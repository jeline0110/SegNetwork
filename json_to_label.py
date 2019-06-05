import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os,shutil
import cv2
import pdb

seg_dir = 'segmention_10'
ans_dir = 'annotations'
img_dir = 'images'
json_dir = 'jsons'


if not os.path.exists(ans_dir):
    os.makedirs(ans_dir)
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
if not os.path.exists(json_dir):
    os.makedirs(json_dir)
   
for file in os.listdir(os.getcwd()):
    if os.path.isfile(file):
        file_name = os.path.basename(file)
        if file_name.split('.')[1] == 'json':
            os.system('labelme_json_to_dataset' + " " + file_name)
        
            label_dir = file_name.split('.')[0] + '_json'
            lable_path = os.path.join(label_dir,'label.png')
            os.rename(lable_path,os.path.join(label_dir,file_name.split('.')[0]+'.png'))
            lable_path = os.path.join(label_dir,file_name.split('.')[0]+'.png')
            if os.path.exists(label_dir):
                shutil.move(lable_path,ans_dir)
                shutil.rmtree(label_dir)
            
            shutil.move(file,json_dir)
        elif file_name.split('.')[1] == 'jpg':
            shutil.move(file,img_dir)
        elif file_name.split('.')[1] == 'py':
            break
'''
for file in os.listdir(ans_dir):
    file_path = os.path.join(ans_dir,file)
    img = Image.open(file_path)
    img = np.array(img)
    img = img.astype(np.uint8)
    img[img==0] = 0
    img[img==1] = 255
    #print(np.unique(img),img.dtype)
    img = Image.fromarray(img)
    img.save(file_path)
'''