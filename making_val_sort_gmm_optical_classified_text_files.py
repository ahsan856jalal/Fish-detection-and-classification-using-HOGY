# @author: ahsan jalal 
# This code takes GT text files, GMM output and optical flow output dir . THe output is text files containing 
# classification results on GMM  & optical combined inputs.






import sys,os,glob
from os.path import join, isfile
import numpy as np
from pylab import *
from PIL import Image
import cv2
#import dlib
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



def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

lib = CDLL("   ~/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.2, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res
net = load_net("~/resnet50.cfg", "~/resnet50_146.weights", 0)
meta = load_meta("~/fish_classification.data")

total_gmm_count=0
total_gt_count=0 
TP=0
FP=0
FP_classify=0
gmm_count=0
num = np.zeros(16) # 17 for uwa
bkg_count=0
vid_counter=0
specie_list= ["vaigiensis",
             "nigrofuscus",
             "clarkii",
             "lununatus",
             "speculum",	
             "trifascialis",
             "chrysura",
             "aruanus",
             "reticulatus",
             "malapterus",
             "kuntee",
             "nigroris",
             "vanicolensis",
             "dickii",
            "scopas",
            "background"] #use uwa classes when applying on uwa dataset



####################################

gmm_optical_main_dir='~/gmm_optical_combined'
gt_main_dir='~/annotated_train_test_comb'
save_main_dir='~/val_sort_gmm_optical_classified_text_files'
saving_dir='~/test_img'
gt_height,gt_width=[640,640]
img_height,img_width=[640,640]

gmm_height,gmm_width=[640,640]
gmm_fol=os.listdir(gmm_main_dir)
vid_counter=0

for video_fol in gmm_fol:
    print('video number {} is in process and video is {}'.format(vid_counter,video_fol))
    vid_counter+=1
    vid_fol_path=join(gmm_main_dir,video_fol)
    os.chdir(vid_fol_path)
    
    gmm_text_files=glob.glob('*.txt')
    for gmm_files in gmm_text_files:
            img_gmm=cv2.imread(gmm_files) # gmm_optical image
            a=open(gmm_files)
            gmm_text=a.readlines()  # gmm_optical_text_file
            img_gt=cv2.imread(join(gt_main_dir,video_fol,gmm_files).split('.txt')[0]+'.png') # gt image
            b=open(join(gt_main_dir,video_fol,gmm_files))
            gt_text=b.readlines()  # gt text file
            gt_count=len(gt_text)
            total_gt_count+=gt_count
            obj_arr = []
            for line_gmm in gmm_text:
            	gmm_count+=1
                total_gmm_count+=1
                # reading each line in gmm-optical result and then check against all ground truths in the respective frame
                line_gmm1 = line_gmm.rstrip()
                coords_gmm=line_gmm1.split(' ')
                # print(coords_gmm-optical)
                
                w_gmm=round(float(coords_gmm[3])*gmm_width)
                h_gmm=round(float(coords_gmm[4])*gmm_height)
                x_gmm=round(float(coords_gmm[1])*gmm_width)
                y_gmm=round(float(coords_gmm[2])*gmm_height)
                x_gmm=int(x_gmm)
                y_gmm=int(y_gmm)
                h_gmm=int(h_gmm)
                w_gmm=int(w_gmm)
                xmin_gmm = x_gmm - w_gmm/2
                ymin_gmm = y_gmm - h_gmm/2
                xmax_gmm = x_gmm + w_gmm/2
                ymax_gmm = y_gmm + h_gmm/2  
                if(xmin_gmm<0):
                    xmin_gmm=0
                if(ymin_gmm<0):
                    ymin_gmm=0
                if(xmax_gmm>gmm_width):
                    xmax_gmm=gmm_width
                if(ymax_gmm>gmm_height):
                    ymax_gmm=gmm_height
                count_gt_line=-1
                match_flag=0
                for line_gt in gt_text:
                    count_gt_line+=1
                    line_gt1 = line_gt.rstrip()
                    coords=line_gt1.split(' ')
                    label_gt=int(coords[0])
                    
                    w_gt=round(float(coords[3])*gt_width)
                    h_gt=round(float(coords[4])*gt_height)
                    x_gt=round(float(coords[1])*gt_width)
                    y_gt=round(float(coords[2])*gt_height)
                    x_gt=int(x_gt)
                    y_gt=int(y_gt)
                    h_gt=int(h_gt)
                    w_gt=int(w_gt)
                    xmin_gt = int(x_gt - w_gt/2)
                    ymin_gt = int(y_gt - h_gt/2)
                    xmax_gt = int(x_gt + w_gt/2)
                    ymax_gt = int(y_gt + h_gt/2)
                    if(xmin_gt<0):
                        xmin_gt=0
                    if(ymin_gt<0):
                        ymin_gt=0
                    if(xmax_gt>gt_width):
                        xmax_gt=gt_width
                    if(ymax_gt>gt_height):
                        ymax_gt=gt_height
                    # now calculating IOMin 
                    
                    xa=max(xmin_gmm,xmin_gt)
                    ya=max(ymin_gmm,ymin_gt)
                    xb=min(xmax_gmm,xmax_gt)
                    yb=min(ymax_gmm,ymax_gt)
                    if(xb>xa and yb>ya):
                        match_flag+=1
                        area_inter=(xb-xa+1)*(yb-ya+1)
                        area_gt=(xmax_gt-xmin_gt+1)*(ymax_gt-ymin_gt+1)
                        area_pred=(xmax_gmm-xmin_gmm+1)*(ymax_gmm-ymin_gmm+1)
                        area_min=min(area_gt,area_pred)
                        area_union=area_pred+area_gt-area_inter
                        
                    #now checking IO over Min area
                    	if(float(area_inter)/area_min>=0.5):
                           TP+=1
                           img_patch=img_gt[ymin_gmm:ymax_gmm,xmin_gmm:xmax_gmm]
                           img_patch = cv2.resize(img_patch.astype('float32'), dsize=(50,50))
                           fish_last_name=specie_list[label_gt]
                           if not os.path.exists(saving_dir):
                              os.makedirs(saving_dir)
                           cv2.imwrite(saving_dir+'/'+ "true_image.png", img_patch)
                           im = load_image(saving_dir+'/'+ "true_image.png", 0, 0)
                           r = classify(net, meta, im)
                           r=r[0]
                           fish_label_det=specie_list.index(r[0])
                           if(fish_label_det==label_gt):
                               TP+=1
                               num[fish_label_det] += 1
                               x=xmin_gmm
                               y=ymin_gmm
                               w=xmax_gmm-xmin_gmm
                               h=ymax_gmm-ymin_gmm
                               x = (x+w/2.0) / img_width
                               y = (y+h/2.0) / img_height
                               w = float(w) / img_width
                               h = float(h) / img_height
                               fish_specie=fish_label_det
                               tmp = [fish_specie, x, y, w, h]
                               obj_arr.append(tmp)
                           else:
                               FP+=1
                               print('False count : {}'.format(FP))

                        else:
                            FP+=1
		                    img_patch=img_gt[ymin_gmm:ymax_gmm,xmin_gmm:xmax_gmm]
		                    img_patch = cv2.resize(img_patch.astype('float32'), dsize=(50,50))
		                    if not os.path.exists(saving_dir):
		                       os.makedirs(saving_dir)
		                    cv2.imwrite(saving_dir+'/'+ "test_image.png", img_patch)
		                    im = load_image(saving_dir+'/'+ "test_image.png", 0, 0)
		                    r = classify(net, meta, im)
		                    r=r[0]
		                    if r[0]=='background' or float(r[1])<0.9:
		                        # cv2.imwrite(save_main_dir+'/'+ r[0]+"_"+str(bkg_count)+"_.png", img_patch)
		                        bkg_count+=1
		                        # print(bkg_count)
		                    else:

		                        FP+=1
#                
    
                if match_flag==0:
                    FP+=1
#			       
	    	xml_content = ""
		for obj in obj_arr:
		    xml_content += "%d %f %f %f %f\n" % (obj[0], obj[1], obj[2], obj[3], obj[4])
		if not os.path.exists(join(save_main_dir,video_fol)):
		    os.makedirs(join(save_main_dir,video_fol))
		f = open(join(save_main_dir,video_fol,gmm_files), "w")
		f.write(xml_content)
		f.close()

print("Total GT detections for the test spllit are: {}".format(total_gt_count))
num
FN=abs(total_gt_count-TP)      
print('True positives are:  ', TP)
print('False Positives are:   ', FP)
print('False Neagatives are:   ', FN)
PR=float(TP)/(TP+FP) 
RE=float(TP)/(TP+FN)
print (' Precision is :    ',PR)     
print (' Recall is :    ',RE )    
F_SCORE=float(2*PR*RE)/(PR+RE)
print (' F-score is :    ', F_SCORE)                                
    

