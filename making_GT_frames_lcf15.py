 # This code is used to save labelled training frames from their respective videos.
import sys,os,glob
import numpy as np
from sklearn import *
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
#import matplotlib.patches as patches
#from PIL import Image, ImageDraw, ImageFont
vid_dir='~/Training_dataset/Videos/'
xml_dir='~/Training_dataset/Ground Truth XML/'# xml are given for lcf-15 dataset
save_img_dir='/home/ahsanjalal/Fishclef/Datasets/Training_dataset/img_pool_retrain1/'
save_lab_dir=save_img_dir
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
            "zebrasoma scopas"]

other_class='others'
other_label=15
all_labels=[]
vid_names=[]
sub_list=array(os.listdir(vid_dir))
#a=np.random.permutation(len(sub_list))
#sub_list=sub_list[a]
img_counter=0


os.chdir(vid_dir)
vid_count=0
for i in range(len(sub_list)):#
    
    print('video number: ',vid_count, ' is in progress')
    vid_count+=1
    image_vid=[]
    cap=cv2.VideoCapture(sub_list[i])
    counter=1
    success,image = cap.read()
    while success:
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_vid.append(image)
        counter+=1
        success,image = cap.read()
    
#    ret,image=cap.read()
    vid_name=sub_list[i].split(".")
    vid_name=vid_name[0]
    xml_name=xml_dir+vid_name+'.xml'
    tree = ET.parse(xml_name)
    root = tree.getroot()
    vid_name_len=len(vid_name)
    vid_name_short=vid_name[vid_name_len-15:vid_name_len]
    other_fish_count=0
    other_frame_number=[]
    other_fish_name=[]
    for child in root:
        frame_id=int(child.attrib['id'])
        if(frame_id<len(image_vid)):
            child_id=0
            
            filename='image_'+str(img_counter)
            
            
            img_counter+=1
            check=0
            for g_child in child:
    #            if check==0:
    #                    all_labels.append(save_img_dir+filename+'.jpg')
                check+=1           
                child_id+=1
                fish_specie=g_child.attrib['fish_species']
                fish_specie=fish_specie.lower()
                if(fish_specie=='chaetodon lununatus'):
                    fish_specie='chaetodon lunulatus'
                    
                
                if fish_specie in specie_list:
    #                try:
    #                    os.stat(save_lab_dir)
    #                except:
    #                    os.mkdir(save_lab_dir) 
                    
        #            fish_name=fish_specie.split(" ")
        #            if len(fish_name>1):
        #                fish_specie=fish_name[1]
                    x=int(g_child.attrib['x'])
                    y=int(g_child.attrib['y'])
                    h=int(g_child.attrib['h'])
                    w=int(g_child.attrib['w'])
                    if x<0:
                        x=0
                    if y<0:
                        y=0
                    frame=image_vid[frame_id]
                    height,width,ch=shape(frame)	
                    frame=imresize(frame,[640,640])
                    
                    mid_x=float(x+x+w)/(2*width)
                    mid_y=float(y+y+h)/(2*height)
                    box_width=float(w)/width
                    box_height=float(h)/height
                    filename=vid_name_short+'_'+'image_'+str(frame_id)
                    scipy.misc.imsave(save_img_dir+filename+'.jpg',frame)
                    a = open(save_lab_dir+filename+'.txt', 'a')
                    fish_lab=specie_list.index(fish_specie)
                    item=str(fish_lab)+' '+str(mid_x)+' '+str(mid_y)+' '+str(box_width)+' '+str(box_height)
                    
                    print>>a, item
                else:
                    other_fish_count+=1
                    other_frame_number.append(frame_id)
                    other_fish_name.append(fish_specie)
                    
        #            fish_name=fish_specie.split(" ")
        #            if len(fish_name>1):
        #                fish_specie=fish_name[1]
                    x=int(g_child.attrib['x'])
                    y=int(g_child.attrib['y'])
                    h=int(g_child.attrib['h'])
                    w=int(g_child.attrib['w'])
                    if x<0:
                        x=0
                    if y<0:
                        y=0
                    frame=image_vid[frame_id]
                    height,width,ch=shape(frame)
                    frame=imresize(frame,[640,640])
                    mid_x=float(x+x+w)/(2*width)
                    mid_y=float(y+y+h)/(2*height)
                    box_width=float(w)/width
                    box_height=float(h)/height
                    fish_specie='other'
                    filename=vid_name_short+'_'+'image_'+str(frame_id)
                    scipy.misc.imsave(save_img_dir+filename+'.jpg',frame)
                    a = open(save_img_dir+filename+'.txt', 'a')
                    fish_lab=other_label
                    item=str(fish_lab)+' '+str(mid_x)+' '+str(mid_y)+' '+str(box_width)+' '+str(box_height)
                    
                    print>>a, item
                    
            a.close()
print('total count for other fish is :   ',other_fish_count)
#
os.chdir('/home/ahsanjalal/Fishclef/Datasets/Test_dataset/video_gmm_results_bkgRatio_08_numframe_200_200_20/sub_0a38e6a322d62fbff33d614c17d8547c#201108200950_2.flv/') 

name='056'     
img=cv2.imread(name+'.png')
height,width,ch=shape(img)
a=open(name+'.txt')
text=a.readlines()
for line in text:
    line = line.rstrip()
    coords=line.split(' ')
    w=round(float(coords[3])*width)
    h=round(float(coords[4])*height)
    x=round(float(coords[1])*width)
    y=round(float(coords[2])*height)
    x=int(x)
    
    y=int(y)
    h=int(h)
    w=int(w)
    xmin = int(x - w/2)
    ymin = int(y - h/2)
    xmax = int(x + w/2)
    ymax = int(y + h/2)
    
    img=cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(255,12,0),2)
imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
#
