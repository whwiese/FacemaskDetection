import torch
import torch.nn as nn
from YOLO.utils import intersection_over_union

class YoloLoss(nn.Module):
    
    def __init__(self, S=7, B=2, C=20):
       
        #we will use MSE loss to measure the success of our predictions
        """
        S = number of boxes images are divided into along each axis (e.g. 7x7 for S=7)
        B = number of bounding boxes each grid space will predict
        C = number of classes in classification problem

        lambda_coord = scalar adjustment to the part of the loss function dealing
            with box coordinates
        lambda_noobj = scalar adjustment to the part of the loss funcion dealing
            with confidence predictions for boxes where no ground truth object
            is present.
        """
        super().__init__()
        self.mse= nn.MSELoss(reduction="sum")

        self.S = S
        self.B = B
        self.C = C

        self.lambda_coord = 5
        self.lambda_noobj = 0.5

    def forward(self, predictions, target):
        #reshape predictions (input predictions are [BATCH_SIZE, S*S*(C+B*5)])
        predictions = predictions.reshape(-1, self.S, self.S, self.C+self.B*5)

        # calculate intersection over union for each B
        # The structure of the last dim of predictions is assumed to be [class_scores...,(x,y,w,h,confidence)*B]
        #    where *B signifies repetition B times 
        ious_list = []
        for box_num in range(self.B):
            box_start = self.C + box_num*5
            ious_list.append(intersection_over_union(predictions[...,box_start:box_start+4],
                target[...,self.C:self.C+4]).unsqueeze(0)) #unsqueeze for concatenation
        
        ious = torch.cat(ious_list)

        #get best_box tensor that signifies which box has highest IoU with ground truth.
        # will have values 0...B signifying the index of the best box
        #target box last dim assumed to have format [one hot classes...,x,y,w,h,exists_object]
        #   where exists_oject = 1 if an object is in the specified grid space, or 0 if there is no object
        iou_maxes, best_box = torch.max(ious, dim=0)
        exists_box = target[..., -1].unsqueeze(3)
        
        box_masks = []
        #generate masks
        for box in range(self.B):
            box_masks.append((best_box==box).float())
        
        # ==================== #
        #    BOX COORDINAES    #
        # ==================== #

        box_preds = torch.zeros(predictions[...,0:4].shape)

        for box in range(self.B):
            box_start = self.C + box*5
            box_preds += box_masks[box]*predictions[..., box_start:box_start+4]

        box_preds = exists_box*box_preds

        box_targets = exists_box*target[..., self.C:self.C+4]

        # take square root of width and height so small deviations are penalized more heavily
        #   in smaller boxes
        # width and height predictions may initially be negative, so we handle that case...
        box_preds[...,2:4] = torch.sign(box_preds[...,2:4])*torch.sqrt(
             torch.abs(box_preds[...,2:4]+1e-8)
        )

        box_targets[...,2:4] = torch.sqrt(box_targets[...,2:4])

        box_coords_loss = self.mse(torch.flatten(box_preds, end_dim=2),
             torch.flatten(box_targets, end_dim=2)
        )
        
        # ================= #
        #    OBJECT LOSS    #
        # ================= #
            
        confidence_preds = torch.zeros(predictions[...,0:1].shape)

        for box in range(self.B):
            box_confidence = self.C + box*5 + 4
            confidence_preds += box_masks[box]*predictions[..., box_confidence:box_confidence+1]

        object_loss = self.mse(
                torch.flatten(exists_box*confidence_preds),
                torch.flatten(exists_box*target[...,self.C+4:self.C+5])
        )

        # ==================== #
        #    NO OBJECT LOSS    #
        # ==================== #

        #assigning loss to all boxes when no object is present in grid space
        no_object_loss = self.mse(
                torch.flatten((1-exists_box) * predictions[..., self.C+4:self.C+5], start_dim=1),
                torch.flatten((1-exists_box) * target[..., self.C+4:self.C+5], start_dim=1)
        )

        for box in range(1,self.B):
            box_confidence = self.C + box*5 + 4
            no_object_loss += self.mse(
                    torch.flatten((1-exists_box) * predictions[..., box_confidence:box_confidence+1], start_dim=1),
                    torch.flatten((1-exists_box) * target[..., self.C+4:self.C+5], start_dim=1)
            )
        # ======================== #
        #    CLASSIFICATION LOSS   #
        # ======================== #

        class_loss = self.mse(
                torch.flatten(exists_box * predictions[...,:self.C], end_dim = -2),
                torch.flatten(exists_box * target[...,:self.C], end_dim = -2),
        )
        
        # ============== # 
        #   FINAL LOSS   #
        # ============== #

        loss = (
                self.lambda_coord * box_coords_loss
                + object_loss
                + self.lambda_noobj * no_object_loss
                + class_loss
        )

        return loss
