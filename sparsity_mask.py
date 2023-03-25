from model import BiSeNet

import torch

import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

def create_sparsity (img, cp):

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = './models/' + cp
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        image = img.resize((512, 512), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

        sparsity_mask = vis_parsing_maps(image, parsing, stride=1)

    return sparsity_mask

def vis_parsing_maps(im, parsing_anno, stride):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
    parsing_mask = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
    sparsing_mask = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
    sparsity_mask = [[1 for j in range(512)] for i in range(512)]

    for i in range(1, 511):
        for j in range(1, 511):
            count = 0
            numb = vis_parsing_anno[i][j]
            if (vis_parsing_anno[i - 1][j - 1] == numb):count += 1
            if (vis_parsing_anno[i][j - 1] == numb):count += 1
            if (vis_parsing_anno[i + 1][j - 1] == numb):count += 1
            if (vis_parsing_anno[i - 1][j] == numb):count += 1
            if (vis_parsing_anno[i + 1][j] == numb):count += 1
            if (vis_parsing_anno[i - 1][j + 1] == numb):count += 1
            if (vis_parsing_anno[i][j + 1] == numb):count += 1
            if (vis_parsing_anno[i + 1][j + 1] == numb):count += 1
            if (count < 7):
                sparsing_mask[i][j] = [0, 0, 0]
                sparsity_mask[i][j] = 0
            else:
               sparsing_mask[i][j] = [255, 255, 255]
               sparsity_mask[i][j] = 1

    #num_of_class = np.max(vis_parsing_anno)

    #for pi in range(1, num_of_class + 1):
    #    index = np.where(vis_parsing_anno == pi)
    #    vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]
    #    parsing_mask[index[0], index[1], :] = part_colors[pi]
    #    if (pi == 2 or pi == 3 or pi == 4 or pi == 5):
    #        sparsing_mask[index[0], index[1], :] = [0, 0, 0]

    #vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    #parsing_mask = parsing_mask.astype(np.uint8)
    #vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    #save_path = './output/mask_photo'
    #cv2.imwrite(save_path[:-5] +'new1.png', parsing_mask)
    #cv2.imwrite(save_path[:-5] +'new2.png', sparsing_mask)
    #cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)

    return sparsity_mask
