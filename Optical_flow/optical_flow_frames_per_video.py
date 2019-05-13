# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 11:48:07 2018

@author: ahsanjalal
""" 

import cv2
import numpy as np
import os
from pylab import *
os.chdir('~/Training_dataset/Videos/')
video_names=os.listdir('~/Training_dataset/Videos/')
save_dir='~/Train_Optical_flow/'
for video_file in video_names:



    video_name=video_file.split('.')[0]
    if not os.path.exists(save_dir+video_file):
        os.makedirs(save_dir+video_file)
    cap = cv2.VideoCapture(video_file)
    ret, frame1 = cap.read()
    fshape = frame1.shape
    fheight = fshape[0]
    fwidth = fshape[1]
    print fwidth , fheight
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#    out = cv2.VideoWriter('/home/ahsanjalal/sub_63ccb5fc2b4233e49ca0fb20a975cc0c#201106051450_1.avi',fourcc, 20.0, (2*fwidth,fheight))
    img_yuv = cv2.cvtColor(frame1, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    frame1 = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)    
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    aa=0
    kernel = np.ones((7,7),np.uint8)
    while(1):
        ret, frame2 = cap.read()
        if(ret==True):
            img_yuv = cv2.cvtColor(frame2, cv2.COLOR_BGR2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            frame2 = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
#            frame2=imresize(frame2,[640,640])\

            next1 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) # current frame
            
            aa+=1
            print(aa)
            flow = cv2.calcOpticalFlowFarneback(prvs,next1, None, 0.95, 10, 15, 3, 5, 1.2, 0) # parameters set for optical flow algorithm used
        
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    #        erosion = cv2.erode(rgb,kernel,iterations = 2)
            opening = cv2.morphologyEx(rgb, cv2.MORPH_OPEN, kernel)
            opening=cv2.cvtColor(opening,cv2.COLOR_BGR2GRAY)
            cv2.imwrite(save_dir+video_file+'/'+"%03d.png" % aa,opening)
    #        opening=cv2.cvtColor(opening,cv2.COLOR_BGR2GRAY)
#            cv2.imshow('frame2',opening)
#            cc=hstack([frame2,opening])
#            out.write(cc)
            
#            k = cv2.waitKey(30) & 0xff
#            if k == 27:
#                break
#            elif k == ord('s'):
#                cv2.imwrite('opticalfb.png',frame2)
#                cv2.imwrite('opticalhsv.png',rgb)
            prvs = next1 # will act as a previous frame
        else:
            break
    cap.release()
    #out.release()
#    cv2.destroyAllWindows()
