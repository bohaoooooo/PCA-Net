import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data
import torchvision.datasets as dates
from torch.autograd import Variable
from torch.nn import functional as F
import shutil
import cv2
from PIL import Image
import tqdm
from einops.einops import rearrange
import math
from torchvision import transforms as transforms1
from torch.optim import lr_scheduler
import cfgs.config as cfg
import dataset.CD_dataset as dates
from torchmetrics import F1Score
import torchvision.transforms as T
import pytorch_ssim

# os.environ['CUDA_LAUNCH_BLOCKING']='1'

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
        
    def forward(self, logits, targets):
        bs = targets.size(0)
        smooth = 1
        
        probs = F.sigmoid(logits)
        m1 = probs.view(bs, -1)
        m2 = targets.contiguous().view(bs, -1)
        
        intersection = (m1*m2)
        
        score = 2. * (intersection.sum(1) + smooth ) / (m1.sum(1) + m2.sum(1) + smooth )
        score = 1 - score.sum() / bs
        
        return score


def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

ab_test_dir = cfg.SAVE_PATH
check_dir(ab_test_dir)

def MIOU(pred, label, smooth=1e-10, n_classes=2) :
    with torch.no_grad() :
        pred = F.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        pred = pred.contiguous().view(-1)

        label = label.contiguous().view(-1)

        iouPerClass = []

        for clas in range(0, n_classes) :
            trueClass = pred == clas
            trueLabel = label == clas

            if trueLabel.long().sum().item() == 0 :
                iouPerClass.append(np.nan)
            else :
                intersect = torch.logical_and(trueClass, trueLabel).sum().float().item()
                union = torch.logical_or(trueClass, trueLabel).sum().float().item()

                iou = (intersect+smooth)/(union+smooth)
                iouPerClass.append(iou)

        return np.nanmean(iouPerClass)

def main():

    train_transform_det = dates.Compose([dates.Scale(cfg.TRANSFROM_SCALES), ])
    val_transform_det = dates.Compose([dates.Scale(cfg.TRANSFROM_SCALES), ])

    train_data = dates.Dataset(cfg.TRAIN_DATA_PATH, cfg.TRAIN_LABEL_PATH,
                               cfg.TRAIN_TXT_PATH, 'train', transform=True,
                               transform_med=train_transform_det)
    train_loader = Data.DataLoader(train_data, batch_size=cfg.BATCH_SIZE,
                                   shuffle=True, num_workers=2, pin_memory=True)
    
    val_data = dates.Dataset(cfg.VAL_DATA_PATH, cfg.VAL_LABEL_PATH,
                             cfg.VAL_TXT_PATH,'val',transform=True,
                             transform_med=val_transform_det)
    val_loader = Data.DataLoader(val_data, batch_size=cfg.BATCH_SIZE,
                                 shuffle=False, num_workers=2, pin_memory=True)
       
    # build model
    import model as models
    device = torch.device("cuda:7")
    model = models.OilSpillDetection()

    model = nn.DataParallel(model, device_ids=cfg.gpu_ids)
    model.to(device)

    # Cross entropy loss
    MaskLoss = torch.nn.CrossEntropyLoss().to(device)
    DiceLoss = SoftDiceLoss().to(device)
    SsimLoss = pytorch_ssim.SSIM().to(device)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.INIT_LEARNING_RATE, momentum=cfg.MOMENTUM, weight_decay=cfg.DECAY)

    # Scheduler, For each 50 epoch, decay 0.1
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    print("train_loader", len(train_loader))
    print("val_loader", len(val_loader))

    loss_pre = 100000
    bestMIoU = 0.
    
    for epoch in range(cfg.MAX_ITER):
        print("epoch", epoch, ", learning rate: ", optimizer.param_groups[0]['lr'])
        model.train()
        
        # Start to train
        for batch_idx, batch in tqdm.tqdm(enumerate(train_loader)):
            step = epoch * len(train_loader) + batch_idx
            img_idx, label_idx, height, width = batch   
            
            img = img_idx.to(device)
            label = label_idx.to(device)
            
            output_map = model(img)
        
            b_num = output_map.shape[0]
            gt = Variable(dates.resize_label(label.data.cpu().numpy(), size=output_map.data.cpu().numpy().shape[2:]).to(device))
            
            loss = MaskLoss(output_map, gt.long()) +  DiceLoss(output_map.argmax(dim=1, keepdim=True), gt.long()) + (1-SsimLoss(output_map.argmax(dim=1, keepdim=True).float(), gt.unsqueeze(1)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (batch_idx) % 100 == 0:
                print(" Epoch [%d/%d] Loss: %.4f " % (epoch, batch_idx, loss.item()))
        loss_total = 0

        # Start to validate
        miou = []
        for batch_idx, batch in tqdm.tqdm(enumerate(val_loader)):
            with torch.no_grad():
                img_idx, label_idx, height, width = batch
                img = Variable(img_idx.to(device))
                label = Variable(label_idx.to(device))
                output_map = model(img)
                gt = Variable(dates.resize_label(label.data.cpu().numpy(), size=output_map.data.cpu().numpy().shape[2:]).to(device))

                loss = MaskLoss(output_map, gt.long()) +  DiceLoss(output_map.argmax(dim=1, keepdim=True), gt.long()) + (1-SsimLoss(output_map.argmax(dim=1, keepdim=True).float(), gt.unsqueeze(1)))
                loss_total = loss_total + loss

                miou.append(MIOU(output_map, gt, smooth=1e-10, n_classes=2))

        scheduler.step()

        print("############################")
        print("loss_total ", loss_total.item())
        print("mIoU ", np.nanmean(miou))

        print('\n')

        if loss_total < loss_pre:  
            loss_pre = loss_total
            torch.save({'state_dict': model.state_dict()}, os.path.join(ab_test_dir, 'model_best.pth'))

        if epoch % 50 == 0:
            torch.save({'state_dict': model.state_dict()}, os.path.join(ab_test_dir, 'model' + str(epoch) + '.pth'))

        if np.nanmean(miou) > bestMIoU :
            bestMIoU = np.nanmean(miou)

    print("best_mIoU ", bestMIoU)
        

if __name__ == '__main__':
    main()