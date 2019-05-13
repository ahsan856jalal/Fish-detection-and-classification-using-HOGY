# HOGY Toolbox for fish detection and categorization
This algorithm detects and classifies fish instances under unconstrained environment using a hybrid of GMM, Optical flow and deep CNN based on YOLO . Preference is given to YOLO during hybridization when results from GMM-optical and YOLO are overlapping
# Making frames from videos

If you want to save GT frames of LCF-15 dataset, use "making_GT_frames_lcf15.py" on the dataset.
For UWA dataset, dataset will be provided upon request {ahsan.jalal@seecs.edu.pk, ahmad.salman@seecs.edu.pk} 
 # GMM Output
 Run "GMM/GMM_frames_per_video.m" to save GMM frames for all videos along with annotated text files. This is written in Matlab

# Optical Flow Output 
Run "Optical_flow/optical_flow_frames_per_video.py" to save Optical flow for the required frames. It is written in python.

# YOLO DNN
For YOLO , clone this repo "https://github.com/AlexeyAB/darknet.git" and make it according to the instructions on "https://github.com/AlexeyAB/darknet" (use libso=1 in Makefile).


# Steps to follow:

- Make training_data list as explained in the aforementioned link
2: edit the yolov3.cfg for lcf-15 and uwa datasets (15 & 16 classes in lcf-15 and uwa datasets respectively)
3: make separate '.names' files for lcf-15 and uwa dataset and put all classes names as mentioned in YOLO instructions
4: make separate '.data' files for each dataset. Copy contents from 'coco.data' file in yolo/cfg directory into each new file and edit classes, train, names and backup variables.
5: For evaluation, you need a pre-trained model on respected datasets. These models will be shared on request {ahsan.jalal@seecs.edu.pk, ahmad.salman@seecs.edu.pk}
6: Once you have the models and test splits, use 'YOLO_DNN/yolo_on_frames.py' to save classification results.

7: Use 'making_gmm_optical_gray_combined_image.py' to combine GMM and optical flow outputs into one 2D frame (green channel to GMM and red to optical flow)
8: ResNet-50 models trained on lcf-15 and UWA datasets are required to classify objects detected by GMM & optical combined. Models will be shared on request {ahsan.jalal@seecs.edu.pk, ahmad.salman@seecs.edu.pk}
9: Once you have the models, use 'making_val_sort_gmm_optical_classified_text_files.py' to save classification results on gmm & optical combined input.
10: Use 'val_sort_gmm_optical_vs_yolo_f_score.py' to calculate f-score for the given dataset using GMM-optical and YOLO classified outputs which will be compared against GTs. Preference is given to YOLO output when the results are overlapping with GMM-optical.

