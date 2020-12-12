import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
import xmltodict

num_to_name = {
    0: "No mask",
    1: "Mask worn properly",
}

def intersection_over_union(box_preds, box_labels, box_format="midpoint"):
    """
    Labels and predictions must have values such that
    x increases to the right and y increases downward in the picture

    Parameters:
        box_preds (tensor): our model's bounding box predictions [BATCH_SIZE,4]
        box_labels (tensor): ground truth bounding box labels [BATCH_SIZE,4]
        box_format (str): fromat that bounding boxes are labeled in
            midpoint - [x,y,w,h] --> x= midpoint x, y= midpoint y, w = width, h=height
            corners - [x1,y1,x2,y2] --> x,y coordinates of upper left, lower right corners
    
    Returns:
        IoU (tensor): Intersection over union for all examples
    """
    if box_format == "midpoint":
        box_preds = midpoint_to_corners(box_preds)
        box_labels = midpoint_to_corners(box_labels)

    #define intersection top left, bottom right corners
    x1 = torch.max(box_preds[...,0:1],box_labels[...,0:1])
    y1 = torch.max(box_preds[...,1:2],box_labels[...,1:2])
    x2 = torch.min(box_preds[...,2:3],box_labels[...,2:3])
    y2 = torch.min(box_preds[...,3:4],box_labels[...,3:4])

    #if there is no intersection x2-x1 or y2-y1 will be less than 0, so we clamp at 0
    intersection = (x2-x1).clamp(0) * (y2-y1).clamp(0)

    pred_area = ( 
        box_preds[...,2:3]-box_preds[...,0:1])*(box_preds[...,3:4]-box_preds[...,1:2])
    label_area = (
        box_labels[...,2:3]-box_labels[...,0:1])*(box_labels[...,3:4]-box_labels[...,1:2])

    union = pred_area + label_area - intersection
    
    #return IoU, 1e-8 constant protects against divide by 0
    return intersection/(union+1e-8)
    
def non_max_supression(bboxes, iou_threshold, prob_threshold, box_format="midpoint"):
    
    """
    Parameters:
        bboxes (list): 
            midpoint: [[class_num, x_mid, y_mid, width, height, prob],
                ...repeat for more bounding boxes]
            corners: 
                [[class_num, x1, y1, x2, y2, prob],...repeat for more bounding boxes]
        iou_threshold (float): iou value between boxes of same class
            at which we will remove lower prob box
        prob_threshold (float): minimum prob a box can have to be considered
        box_format (str): 'midpoint' or 'corners'. Designates coordinate representation
            of the bounding boxes. See formats under bboxes above.

    Returns:
        bboxes_after_nms (list): post-nms bounding boxes
    """
    
    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[-1] > prob_threshold]
    bboxes = sorted(bboxes, key=lambda x:x[-1], reverse=True)

    bboxes_after_nms = []

    while bboxes:
        highest_prob = bboxes.pop(0)

        bboxes = [box for box in bboxes
                    if box[0] != highest_prob[0]
                    or intersection_over_union(
                        torch.tensor(box[1:-1]),
                        torch.tensor(highest_prob[1:-1]),
                        box_format
                        )
                        < iou_threshold
                    ]
        
        bboxes_after_nms.append(highest_prob)

    return bboxes_after_nms

#get mean average precision over multiple IoU thresholds
def mean_average_precision(pred_boxes, true_boxes, num_classes=20, box_format="midpoint", 
        start_threshold=0.05, end_threshold=0.95, step_size=0.05):
    
    map_total = 0.0
    num_thresholds = 0
    for iou_threshold in np.arange(start_threshold, end_threshold+1e-8, step_size):
        map_total += single_map(pred_boxes, true_boxes, iou_threshold, num_classes, box_format)
        num_thresholds += 1

    return map_total / num_thresholds




def single_map(
        pred_boxes, true_boxes, iou_threshold=0.5, num_classes=20, box_format="midpoint"):
    """
    Parameters:
        pred_boxes (list): [[image_index, class_pred, x_mid, y_mid, w, h, prob],[],[],...]
            where each sublist represents a bounding box. Boxes may also be represented
            in "corners" form (x1,y1,x2,y2)
        true_boxes (list): Ground truth bounding boxes. Same format as pred boxes.
        iou_threshold (float): Minimum IoU value at which we will consider a prediction to be correct.
        num_classes (int): Number of classes our detector may assign an object to.
        box_format (str): "midpoint" or "corners" specifies box coordinate representation.

    Returns:
        mAP (float): mean average precision across all classes for a specific IoU threshold.
    """
    total_average_precision = 0.0
    
    #to protect against divide by zero errors later
    epsilon = 1e-8

    num_gt_classes = 0

    for c in range(num_classes):
        predictions = []
        ground_truths = []
        
        
        #collect ground truth boxes that belong to class c
        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        total_gt_bboxes = len(ground_truths)

        #skip class if there are no ground truth bounding boxes assigned to it 
        if total_gt_bboxes == 0:
            continue
        
        # collect detections that belong to class c
        for pred_box in pred_boxes:
            if pred_box[1] == c:
                predictions.append(pred_box)

        # make dictionary mapping image_index to the number of times 
        # it appears in ground_truths. This gives us the number of
        # ground truth instances of class c present in each image
        gt_bbox_counter = Counter([bbox[0] for bbox in ground_truths])

        # reassign values of gt_bbox_counter to a torch tensor
        # with length equal to the number of ground truth
        # bounding boxes in that image. This will help us later
        # to determie which gt boxes have already been predicted
        for image_index, num_bboxes in gt_bbox_counter.items():
            gt_bbox_counter[image_index] = torch.zeros(num_bboxes)
        
        #sort predictions by probability in descending order
        predictions.sort(key=lambda x: x[-1], reverse=True)
        true_positives = torch.zeros((len(predictions)))
        false_positives = torch.zeros((len(predictions)))

        for detection_index, prediction in enumerate(predictions):
            # get ground truth bounding boxes for the image
            # corresponding to this detection
            ground_truth_bboxes= [bbox for bbox in ground_truths if bbox[0] == prediction[0]]
            num_gt_bboxes = len(ground_truth_bboxes)
            best_iou = 0

            for gt_index, gt in enumerate(ground_truth_bboxes):
                iou = intersection_over_union(
                        torch.tensor(prediction[2:-1]),
                        torch.tensor(gt[2:-1]),
                        box_format = box_format
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = gt_index
            
            # check if best_iou is above our true positive threshold. 
            # must also check if another prediction has already been made for this ground truth
            # if so this example is false positive
            if best_iou > iou_threshold and gt_bbox_counter[prediction[0]][best_gt_index] == 0:
                true_positives[detection_index] = 1
                gt_bbox_counter[prediction[0]][best_gt_index] == 1
            else:
                false_positives[detection_index] = 1
        
        # calculate average precision for this class
        # by numerically integrating y(x) where 
        # y = cumulative precisiomn and x = cumulative recall

        tp_cumsum = torch.cumsum(true_positives, dim=0)
        fp_cumsum = torch.cumsum(false_positives, dim=0)

        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + epsilon)
        recalls = tp_cumsum / (total_gt_bboxes + epsilon)
        
        #add point (0,1) for numerical integration
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        total_average_precision += torch.trapz(precisions, recalls)
        num_gt_classes += 1

    return total_average_precision / num_gt_classes        
        
def corners_to_midpoint(box_corners):
    """
    Parameters:
        box_corners (tensor) [...,4]: Last dim = [x1,y1,x2,y2] 

    Returns:
        box_midpoint (tensor [...,4]): Last dim = [x_mid,y_mid,width,height]
    """
    box_midpoint = torch.zeros(box_corners.shape)

    box_midpoint[...,0] = (box_corners[...,2] + box_corners[...,0]) / 2
    box_midpoint[...,1] = (box_corners[...,3] + box_corners[...,1]) / 2
    box_midpoint[...,2] = (box_corners[...,2] - box_corners[...,0])
    box_midpoint[...,3] = (box_corners[...,3] - box_corners[...,1])
    
    return box_midpoint

def midpoint_to_corners(box_midpoint):
    """
    Parameters:
        box_midpoint (tensor, [...,4]): Last dim = [x_mid,y_mid,width,height] 

    Returns:
        box_corners (tensor) [...,4]: Last dim = [x1,y1,x2,y2]
    """
    box_corners = torch.zeros(box_midpoint.shape)
    box_corners[...,0] = box_midpoint[...,0] - box_midpoint[...,2]/2
    box_corners[...,1] = box_midpoint[...,1] - box_midpoint[...,3]/2
    box_corners[...,2] = box_midpoint[...,0] + box_midpoint[...,2]/2
    box_corners[...,3] = box_midpoint[...,1] + box_midpoint[...,3]/2
    
    return box_corners

def get_bboxes(loader, model, iou_threshold, prob_threshold, S=7,
        C=20, pred_format="cells", box_format="midpoint", 
        device="cpu", mode="all"
    ):
    """
    runs forward pass of model,
    returns predicted bboxes that have predicted probability
    above prob_threshold and ground truth bboxes

    mode: set mode to "all" to get all bounding boxes
        set mode to "batch" to get bounding boxes from a random batch
    """
    all_pred_bboxes = []
    all_gt_bboxes = []
    
    #set model to evaluation mode
    model.eval()
    train_index = 0

    if mode == "batch":
        batches =  enumerate([next(iter(loader))]) 
    else:
        batches = enumerate(loader)

    for batch_index, (x, labels) in batches: 
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        predictions = predictions.reshape(predictions.shape[0],S,S,-1)
        batch_size = x.shape[0]
        gt_bboxes = cells_to_full_image(labels, S, C)
        pred_bboxes = cells_to_full_image(predictions, S, C)

        for example in range(batch_size):
            #nms input boxes have form [[class_num, x_mid, y_mid, width, height, prob],[],...]
            nms_bboxes = non_max_supression(
                pred_bboxes[example],
                iou_threshold=iou_threshold,
                prob_threshold=prob_threshold,
                box_format=box_format
            )

            for nms_bbox in nms_bboxes:
                all_pred_bboxes.append([train_index] + nms_bbox)

            for gt_bbox in gt_bboxes[example]:
                all_gt_bboxes.append([train_index] + gt_bbox)

            train_index += 1

    model.train()
    return all_pred_bboxes, all_gt_bboxes

def cells_to_full_image(model_out, S, num_classes):
    """
    PARAMETERS:
        model_out (tensor): [batch_size, S, S, num_classes + B*5]
            where B = number of bounding boxes per cell
        num_classes (int): number of classes in classification problem
        S (int): number of cells along each dimension of YOLO grid
    RETURNS:
        box_list: list of [[class_num, x_mid, y_mid, width, height, prob], [], ...]
            where each sub box is a separate prediction or ground truth label
    """
    num_boxes = (model_out.shape[-1]-num_classes)//5

    box_list = []
    for box_index in range(model_out.shape[0]):
        example_boxes = []
        for i in range(S):
            for j in range(S):
                for box in range(num_boxes):
                    prob = model_out[box_index,i,j,num_classes+(box+1)*5-1].item()
                    if prob > 0.001:
                        x = (1/S)*(model_out[box_index,i,j,num_classes+box*5].item() + i)
                        y = (1/S)*(model_out[box_index,i,j,num_classes+box*5+1].item() + j)
                        w = (1/S)*(model_out[box_index,i,j,num_classes+box*5+2].item())
                        h = (1/S)*(model_out[box_index,i,j,num_classes+box*5+3].item())
                        pred_class = torch.argmax(model_out[box_index,i,j,:num_classes]).item()
                        example_boxes.append([pred_class, x, y, w, h, prob])
        box_list.append(example_boxes)  

    return box_list

def plot_detections(image, labels):
    """
    Plots object bounding boxes and class labels over image.
    labels is a list of lists where each sublist is a label for
        an individual bounding box.
    individual labels are in form 
        [class_label, x_mid, y_mid, width, height, prob].
    bbox values are relative to image size (range from 0 to 1).
    """
    img = np.array(image)
    img_height, img_width, _ = img.shape
    
    #create figure
    fig, ax = plt.subplots(1)
    #display image
    ax.imshow(img)

    title_string = "Detected: "
    
    for label in labels:
        bbox = label[1:-1]
        class_label = label[0]
        prob = label[-1]
        
        title_string += (" "+num_to_name[class_label]+",")
        
        # Rectangle function plots rectangles in form 
        # [x_upper_left, y_upper_left, widht, height]
        # so we convert our bbox to this form
        x_ul = bbox[0] - bbox[2] / 2
        y_ul = bbox[1] - bbox[3] / 2
        
        #draw bboxes
        rect = patches.Rectangle(
            (x_ul*img_width, y_ul*img_height),
            bbox[2] * img_width,
            bbox[3] * img_height,
            linewidth=2,
            edgecolor="g",
            facecolor="none",
        ) 
        ax.add_patch(rect)
    
    #plot formatting
    title_string = title_string[:-1]
    plt.title(title_string)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    plt.show()

def parse_xml(label_path):
    
    x = xmltodict.parse(open(label_path , 'rb'))
    item_list = x['annotation']['object']
    
    # when image has only one bounding box
    if not isinstance(item_list, list):
        item_list = [item_list]
        
    result = []

    img_width, img_height = (int(x['annotation']['size']['width']), 
            int(x['annotation']['size']['height']))
    
    for item in item_list:
        name = item['name']
        if name == "with_mask":
            class_id = 1
        else:
            class_id = 0

        bndbox = [int(item['bndbox']['xmin'])/float(img_width),
                int(item['bndbox']['ymin'])/float(img_height),
                int(item['bndbox']['xmax'])/float(img_width),
                int(item['bndbox']['ymax'])/float(img_height)
        ]       

        bndbox = corners_to_midpoint(torch.tensor(bndbox)).tolist()
        result.append([class_id]+bndbox)
    
    return result

def ceildiv(a, b):
    """
    performs ceiling integer division
    """
    return -(-a // b)
