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
from PIL import Image
import matplotlib.pyplot as plt
import tqdm
from torchvision.utils import save_image
import cv2

import glob    
import torchvision.transforms.functional as TF
import torchvision.transforms as T

import model as models


if __name__ == '__main__' :
    # dataset
    data_name = 'ESA_SAR'     

    # image              
    img_folder = './oilspill/patch/test/images_clear'
    gt_folder = './oilspill/patch/test/labels_1D'

    # load model and weight
    pretrain_deeplab_path = "./output/ESA_SAR/model_best.pth"

    device = torch.device("cuda:0")
    model = models.OilSpillDetection()
    model = nn.DataParallel(model, device_ids=[0])
    checkpoint = torch.load(pretrain_deeplab_path)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    
    for img_file in os.listdir(img_folder) :
        gt_file = img_file[:-7] + 'mask.png'

        if img_file[-4:] == '.jpg' :
            img = Image.open(os.path.join(img_folder, img_file))
            label = Image.open(os.path.join(gt_folder, gt_file)).resize((160, 160))

            width, height = img.size

            inputImg = img.resize((160, 160))
            inputImg = TF.to_tensor(inputImg)

            if data_name == 'ESA_SAR' :
                img = TF.normalize(img, mean=[0, 0, 0], std=[1, 1, 1])

            inputs = Variable(inputImg.to(device).unsqueeze(0))

            # model
            output_map = model(inputs)
            output_map = output_map.detach()
            output_show = output_map.argmax(dim=1, keepdim=True)

            label = np.array(label.convert('L'))
            label[label>0] = 1

            output_cal = output_show[0].cpu().numpy().astype(np.uint8)
    
            transform = T.ToPILImage()

            output_show = transform(255*output_cal.transpose(1, 2, 0))
            output_show = output_show.resize((width, height), Image.BILINEAR)
            output_show.save("./ESA_SAR_output/" + img_file[:-4] + ".png")