import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from torch.utils.data import (DataLoader, random_split)
from YOLO.model import (YOLOv1, YOLOv2_lite)
from YOLO.dataset import YOLOMaskDataset
from YOLO.utils import (
        intersection_over_union,
        non_max_supression,
        mean_average_precision,
        single_map,
        get_bboxes,
)
from YOLO.loss import YoloLoss
import time

seed = 123
torch.manual_seed(seed)

#YOLO hyperparameters
GRID_SIZE = 13
NUM_BOXES = 2
NUM_CLASSES = 2

#other hyperparameters
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 10
WEIGHT_DECAY = 0
EPOCHS = 200
NUM_WORKERS = 2
PIN_MEMORY = False
LOAD_MODEL = False 
DROP_LAST = False
TRAINING_DATA = "data/facemask_852_examples.csv"
SAVE_MODEL_PATH = "saved_models/masks1.pt"
LOAD_MODEL_PATH = "saved_models/masks1.pt"
SAVE_EVERY = 5
IMG_DIR = "data/images"
LABEL_DIR = "data/annotations"

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, labels):
        for t in self.transforms:
            img, labels = t(img), labels

        return img, labels

transform = Compose([transforms.Resize((416,416)), transforms.ToTensor()])

def train_fn(loader_train, model, optimizer, loss_fn):
    
    mean_loss = []

    for batch_index, (x, y) in enumerate(loader_train):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Mean loss: %f"%(sum(mean_loss)/len(mean_loss)))

def main():
    model = YOLOv2_lite(grid_size=GRID_SIZE, 
            num_boxes=NUM_BOXES, num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss(S=GRID_SIZE, B=NUM_BOXES, C=NUM_CLASSES)

    if LOAD_MODEL:
        model.load_state_dict(torch.load(LOAD_MODEL_PATH))

    mask_dataset = YOLOMaskDataset(
            TRAINING_DATA, transform=transform, 
            img_dir=IMG_DIR, label_dir=LABEL_DIR,
            S=GRID_SIZE, B=NUM_BOXES, C=NUM_CLASSES,
    )

    train_dataset, val_dataset = random_split(mask_dataset, [750,102], 
            generator=torch.Generator().manual_seed(42)) 
    
    train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            #num_workers=NUM_WORKERS,
            #pin_memory=PIN_MEMORY,
            shuffle=True,
            drop_last=False,
    )
    
    val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=BATCH_SIZE,
            #num_workers=NUM_WORKERS,
            #pin_memory=PIN_MEMORY,
            shuffle=True,
            drop_last=True,
    )

    epochs_passed = 0

    for epoch in range(EPOCHS):

        if epochs_passed%5 == 0:
            train_pred_boxes, train_target_boxes = get_bboxes(
                    train_loader, model, iou_threshold=0.5,
                    prob_threshold=0.4, S=GRID_SIZE, 
                    C=NUM_CLASSES, mode = "batch",
            )

            val_pred_boxes, val_target_boxes = get_bboxes(
                    val_loader, model, iou_threshold=0.5, 
                    prob_threshold=0.4, S=GRID_SIZE,
                    C=NUM_CLASSES, mode = "batch",
            )
            # map function takes predicted boxes and ground truth
            # boxes in form [[],[],[],...] where each sublist is a bounding box
            # of form [image_index, class_pred, x_mid, y_mid, w, h, prob]

            train_mAP = single_map(
                    train_pred_boxes, train_target_boxes, 
                    box_format="midpoint", num_classes=NUM_CLASSES,
            )

            val_mAP = single_map(
                    val_pred_boxes, val_target_boxes, 
                    box_format="midpoint", num_classes=NUM_CLASSES,
            )

            print("Train mAP: %f"%(train_mAP))
            print("Val mAP: %f"%(val_mAP))

        if epochs_passed%SAVE_EVERY == 0:
            save_path = SAVE_MODEL_PATH + ("%d_e"%(epochs_passed))
            torch.save(model.state_dict(),save_path)
            print("Trained for %d epochs"%(epochs_passed))

        train_fn(train_loader, model, optimizer, loss_fn)
        epochs_passed += 1

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Training time: %f seconds"%(time.time()-start_time))
