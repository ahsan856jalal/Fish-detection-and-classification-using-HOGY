# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 16:56:16 2018

@author: ahsanjalal
"""


import sys,os,glob
from os.path import join, isfile
import numpy as np
from pylab import *
from PIL import Image
import cv2
import dlib
from scipy.misc import imresize
from statistics import mode
from tempfile import TemporaryFile
from collections import Counter
import numpy
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
#from rgb2gray import rgb2gray
import lxml.etree
import scipy.misc
from natsort import natsorted, ns
import xml.etree.ElementTree as ET
from shutil import copytree
import matplotlib.pyplot as plt
import glob
import os
import PIL
from ctypes import *
import math
import random

bkg_count=0
# main directories
gt_dir='~/annotated_frames'
gmm_results='~/gmm_output'
optical_results='~/Optical_flow'
save_main_dir='~/no_gray_gmm_optical_mixed'
total_gt_count=0
gt_fol=os.listdir(gt_dir)
TP=0
FP=0
gmm_count=0

num = np.zeros(16) # 17 for UWA dataset
vid_counter=0
specie_list= ["abudefduf vaigiensis",
             "acanthurus nigrofuscus",
             "amphiprion clarkii",
             "chaetodon lununatus",
             "chaetodon speculum",  
             "chaetodon trifascialis",
             "chromis chrysura",
             "dascyllus aruanus",
             "dascyllus reticulatus",
             "hemigumnus malapterus",
             "myripristis kuntee",
             "neoglyphidodon nigroris",
             "pempheris vanicolensis",
             "plectrogly-phidodon dickii",
            "zebrasoma scopas",
            "Background"] # use UWA names for UWA dataset
for video_fol in gt_fol:
    print('video number {} is in process and video is {}'.format(vid_counter,video_fol))
    vid_counter+=1
    vid_fol_path=join(gt_dir,video_fol)
    os.chdir(vid_fol_path)
    video_name=video_fol.split('.flv')[0]
    gt_text_files=glob.glob('*.txt')
    gt_height,gt_width=[640,640]
    gmm_height,gmm_width=[640,640]
    for gt_files in gt_text_files:
        img_gt=cv2.imread(gt_files.split('.')[0]+'.png')
        a=open(gt_files)
        gt_text=a.readlines()
        gt_count=len(gt_text)
        total_gt_count+=gt_count
        
    # reading infofromn the ground truth files
    # 'del list[index]' to remove the specific line from the list
        
        if os.path.isfile(join(gmm_results,video_fol,gt_files).split('.txt')[0]+'.png'):
            
#            gmm_text=open(join(gmm_results,video_fol,gt_files))
            img_gmm=cv2.imread(join(gmm_results,video_fol,gt_files).split('.txt')[0]+'.png')
            img_optical=cv2.imread(join(optical_results,video_fol,gt_files).split('.txt')[0]+'.png')
            img_optical=imresize(img_optical,[640,640])
            img_gt_gray=cv2.cvtColor(img_gt,cv2.COLOR_BGR2GRAY)
#            img_gt[:,:,0]=img_gt_gray
            img_gt[:,:,0]=0
            img_gt[:,:,1]=img_gmm[:,:,0]
            img_gt[:,:,2]=img_optical[:,:,0]
            if not os.path.exists(join(save_main_dir,video_fol)):
                os.makedirs(join(save_main_dir,video_fol))
            cv2.imwrite(join(save_main_dir,video_fol,gt_files).split('.txt')[0]+'.png',img_gt)
        else:
            img_gmm=np.zeros(shape=[640,640])
            if os.path.isfile(join(optical_results,video_fol,gt_files).split('.txt')[0]+'.png'):
                img_optical=cv2.imread(join(optical_results,video_fol,gt_files).split('.txt')[0]+'.png')
                img_optical=imresize(img_optical,[640,640])
            else:
                img_optical=np.zeros(shape=[640,640,3])
            
            img_gt_gray=cv2.cvtColor(img_gt,cv2.COLOR_BGR2GRAY)
            img_gt[:,:,0]=0  # no gray channel             
#            img_gt[:,:,0]=img_gt_gray
            img_gt[:,:,1]=img_gmm
            img_gt[:,:,2]=img_optical[:,:,0]
            if not os.path.exists(join(save_main_dir,video_fol)):
                os.makedirs(join(save_main_dir,video_fol))
            cv2.imwrite(join(save_main_dir,video_fol,gt_files).split('.txt')[0]+'.png',img_gt)
#            text_gmm=gmm_text.readlines()














