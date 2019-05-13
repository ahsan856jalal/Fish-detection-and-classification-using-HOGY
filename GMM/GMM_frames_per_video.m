% author: ahsanjalal


% close all;
% clear all;
main_dir='~/Train_video_gmm_results_bkgRatio_07_numframe_250_ga_20_sz_200_disk/';%path to saving dir
video_dir='~/Training_dataset/Videos/'; % path to the video directory
chdir(video_dir)

video_name_list=dir('*.flv');
for vids=1:length(video_name_list)
    video_name=video_name_list(vids).name;
    fprintf('video number %d is in process \n',vids)
    
    
    
    
% fish_info={};
% se = strel('square', 3);
chdir(main_dir)
% opFolder = fullfile(cd,video_name);
% if ~exist(opFolder, 'dir')
%     mkdir(opFolder);
% end

opfolder = fullfile(cd, video_name);
if ~exist(opfolder, 'dir')
    mkdir(opfolder);
end


foregroundDetector = vision.ForegroundDetector('NumGaussians', 20, ...
    'NumTrainingFrames', 250, 'MinimumBackgroundRatio', 0.7); % These are for fishclef dataset
% videoReader = vision.VideoFileReader(video_name);
% fRate = videoReader.info.VideoFrameRate;
% frame_rate = ['frmae rate of the input video is ',num2str(fRate), 'fps'];
% disp(frame_rate);
 
% total_Frames = 0;
% while ~isDone(videoReader)
%   I = step(videoReader);
%   total_Frames = total_Frames+1;
% end
% X = ['total num of frames in the video are ',num2str(total_Frames),];
% disp(X)
numFramesWritten = 0;
blobAnalysis = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
    'AreaOutputPort', true, 'CentroidOutputPort', true, ...
    'MinimumBlobArea', 200);


%%%%%%%%%%%%%%%% create an empty array of tracks%%%%%%%%%%%%%%%%%%%%

tracks = struct('id', {},'bbox', {},'kalmanFilter', {},'age', {}, ...
            'totalVisibleCount', {},'consecutiveInvisibleCount', {});
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for n = 1:total_Frames,
%     frames(:,:,:,n) = step(videoReader);
% end
i=-1;
chdir(video_dir);
videoReader = vision.VideoFileReader(video_name);
% videoReader = vision.VideoFileReader('sub_a0eeffe39a2c341c6899839ee91fb59f#201103100600_4.flv');

while ~isDone(videoReader)
    i=i+1;
   present_fish=[];
%    frames(:,:,:,i) = step(videoReader);
%     frame=imsharpen(step(videoReader));
    frame=step(videoReader);
    frame=imresize(frame,[640,640]); % resolution used for lcf-15 data
    frame=imadjust(frame,[],[],1.5);
%     frame=double(frame)/255;
%     frame_hsv=rgb2hsv(frame);
%     frame_ycbcr=rgb2ycbcr(uint8(frame));
%     C_ycbcr_bw=im2bw(frame);  
    foreground = step(foregroundDetector, frame);
    % Apply morphological operations to remove noise and fill in holes.
        filteredForeground=foreground;
        filteredForeground= imopen(filteredForeground, strel('disk',3));  % structuring ele for morph opening
        filteredForeground = imclose(filteredForeground, strel('disk',5)); % structuring ele for morph closing
        
%         test=imfill(filteredForeground, 'holes');
%         filteredForeground= imfill(filteredForeground, 'holes');
        
 %%%extracting centroid and anre of the blob using regionprops function%%%%       
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
        
  [area,centroid,bbox] = step(blobAnalysis, filteredForeground);                          % Perform blob analysis to find connected components.       
  

 %%%calculating area and centroid of the bounding box instead of blob%%%%%%        
       
       % for i=1:size(bbox, 1)
       %centroid(i,:) = [ bbox(i,1)+bbox(i,3)/2 ; bbox(i,2)+bbox(i,4)/2 ];
       %area(i,1) =  bbox(i,3)*bbox(i,4);
       %z=['centroid value for the bounding box is',num2str(area(i,1)),]
       %disp(z)
       %end
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%     result = insertShape(frame, 'Rectangle', [bbox;bbox], 'Color', 'red');
%     numFishes = size(bbox, 1);
%     result = insertText(result, [10 10], numFishes, 'BoxOpacity', 0.6, ...
%     'FontSize', 25);%   
if(size(bbox,1)>0)
opBaseFileName = sprintf('%3.3d.png', i);
textfilename = sprintf('%3.3d.txt', i);
opFullFileName = fullfile(opfolder, opBaseFileName);
opFullFiletext = fullfile(opfolder, textfilename);
test_image=zeros(size(filteredForeground));


    
[img_height,img_width,ch]=size(frame);
for d=1:size(bbox,1)
x=bbox(d,1);
y=bbox(d,2);
w=bbox(d,3);
h=bbox(d,4);

if(y+h> img_height && x+w > img_width)
    test_image(y:img_height,x:img_width)=filteredForeground(y:img_height,x:img_width);
    
elseif(y+h > img_height)
    test_image(y:img_height,x:x+w)=filteredForeground(y:img_height,x:x+w);
elseif(x+w > img_width)
    test_image(y:y+h,x:img_width)=filteredForeground(y:y+h,x:img_width);
  
else
    test_image(y:y+h,x:x+w)=filteredForeground(y:y+h,x:x+w);
end
       

    x = double(x+w/2.0) / img_width;
    y = double(y+h/2.0) / img_height;
    w = double(w) / img_width;
    h = double(h) / img_height;
%     tmp = ['0', int2str(x), int2str(y), int2str(w), int2str(h)];
    fileID = fopen(opFullFiletext,'a');
    fprintf(fileID, '%d %f %f %f %f\n', 0, x,y,w,h);
    fclose(fileID);

end
imwrite(test_image, opFullFileName, 'png');

else
    textfilename = sprintf('%3.3d.txt', i);
    opFullFiletext = fullfile(opfolder, textfilename);
    fileID = fopen(opFullFiletext,'a');
    fclose(fileID);

end

end


end





