import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
import tifffile as tiff
import cv2
from PIL import Image


path = "DRIVE/training/1st_manual/21_manual1.gif"
large_image_stack = tiff.imread('DRIVE/training/images/21_training.tif')
large_mask_stack = cv2.imread('DRIVE/training/1st_manual/21_manual1.gif')

patches_img = patchify(large_image_stack, (16,16,3), step=16)
#patches_mask = patchify(large_mask_stack, (128,128,3), step=128)

for i in range(patches_img.shape[0]):
    for j in range(patches_img.shape[1]):
        single_patch_img = patches_img[i,j,:,:]
        tiff.imwrite('patch/images/' + 'image_'+'_'+ str(i)+str(j) + ".tif", single_patch_img)
#
# for i in range(patches_mask.shape[0]):
#     for j in range(patches_mask.shape[1]):
#         single_patch_mask = patches_mask[i,j,:,:]
#         tiff.imwrite('patch/masks/' + 'image_'+'_'+ str(i)+str(j) + ".tif", single_patch_mask)