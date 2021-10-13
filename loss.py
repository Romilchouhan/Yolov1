import torch
import torch.nn as nn
import torchvision
from torchvision.ops import box_iou

class YoloLoss(nn.Module):
    """
    S -> number of grids per image
    B -> number of bounding boxes per grid
    C -> number of classes
    """

    def __init__(self, S=7, B=2, C=20) -> None:
        super().__init__()
        self.S = S
        self.B = B
        self.C = C

        self.mse = nn.MSELoss(reduction="sum")

        self.lamba_noobj = 0.5
        self.lamba_coord = 4

    def forward(self, predictions, target):
        # predictions are shaped (BATCH_SIZE, S*S*(C+B*5))
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # Calculate IoU for the two predicted bounding boxes with target boxes
        iou_b1 = box_iou(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = box_iou(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        
        # Take the box with highest IoU 
        # Note that bestbox will be indices of 0, 1 for which bbox was best
        # if first bbox has high IoU then bestbox = 0 
        # and if second bbox has high IoU then bestbox = 1
        iou_maxes, bestbox = torch.max(ious, dim=0)  # return of torch.max is (values, indices)
        exists_box = target[..., 20].unsqueeze   # identity_obj_i : represent the presence of a object in the bbox
                                                 # there may be background in which case the value will be 0

        
        """BOX-COORDINATE LOSS"""
        
        # calculate the loss only if there is an object in the bbox 
        # i.e, exists_box = 1 (no background)
        box_predictions = exists_box * (
                bestbox * predictions[26:30] + (1 - bestbox) * predictions[21:25]
                )
        box_target = exists_box * target[21:25]

        # take the square root of the width and height of 
        # the predicted and target bounding boxes
        box_predictions = torch.sign(box_predictions[..., 2:4]) * torch.abs(torch.sqrt(box_predictions[..., 2:4] + 1e-6))
        box_target = torch.sqrt(box_target[..., 2:4])
        
        box_loss = self.mse(
                torch.flatten(box_predictions, end_dim=-2),
                torch.flatten(box_target, end_dim=-2)
                )
        
        """OBJECT LOSS"""

        pred_box = exists_box * (
                bestbox * predictions[25:26] + (1 - bestbox) * predictions[20:21]
                )
        obj_loss = self.mse(
                torch.flatten(pred_box),
                torch.flatten(target[..., 20:21])
                )

        """NO OBJECT LOSS"""

        # (N, S, S, 1) -> (N, S*S)
        no_obj_loss = self.mse(
                torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
                torch.flatten((1 - exists_box) * target[20:21], start_dim=1)
                )

        no_obj_loss += self.mse(
                torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
                torch.flatten((1 - exists_box) * target[20:21], start_dim=1)
                )
        
        """CLASS LOSS"""

        # (N, S, S, 20) -> (N*S*S, 20)
        class_loss = self.mse(
                torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
                torch.flatten(exists_box * target[..., :20], end_dim=-2)
                )

        loss = self.lamba_coord * box_loss + obj_loss + self.lamba_noobj * no_obj_loss + class_loss
        return loss


