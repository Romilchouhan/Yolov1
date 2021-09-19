import torch
from torch.optim import optimizer
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch.optim as optim 
import tqdm as tqdm   # Nice progress bar
from model import Yolov1
from loss import YoloLoss
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from utils import (
        mean_average_precision,
        cellboxes_to_boxes,
        get_bboxes,
        plot_bboxes,
        save_checkpoint,
        load_checkpoint
        )

seed = 32
torch.manual_seed(seed)

# Hyperparameters 
LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8  # 64 in original but I don't have that much compute power
WEIGHT_DECAY = 0
EPOCHS = 1  # Just to check
NUM_WORKERS = 2  # Set according to your gpu
PIN_MEMORY = True
LOAD_MODEL = False 
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes



transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (image, target) in enumerate(loop):
        image = image.to(device = DEVICE)
        target = target.to(device = DEVICE)

        # forward
        score = Yolov1(image)
        loss = model(score, target)
        mean_loss.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent step
        optimizer.step()
        
        # update the progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss: {mean_loss.sum()/len(mean_loss)}")

def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    loss_fn = YoloLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, 
                            weight_decay=WEIGHT_DECAY)
    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)
    

  
    train_data = datasets.VOCDetection('./data/VOC_trainval_data', '2007', 
            image_set='trainval', download=True,
            transform=transform)
    test_data = datasets.VOCDetection('./data/VOC_test_data', '2007',
            image_set='test', download=True,
             transform=transform) 
 
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, 
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY, shuffle=True)


    for epoch in range(EPOCHS):

        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
            )

        mean_avg_precision = mean_average_precision(
                pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
                )
    
    train_fn(train_loader, model, optimizer, loss_fn)

if __name__ == "__main__":
    main()







