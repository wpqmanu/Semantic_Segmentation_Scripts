clear all; close all;
% This function is used to generate training/validation labels from the
% colorful groundtruth to grayscale ground truth that is mandatory for the
% caffe interface in DeepLab_V2.

%dataset = 'VOC2012';
%orig_folder = fullfile('..', dataset, 'SegmentationClassAug_Visualization');
%save_folder = ['../', dataset, '/SegmentationClassAug'];

%orig_folder = '/rmt/work/deeplabel/exper/voc12/res/erode_gt/post_densecrf_W41_XStd33_RStd4_PosW3_PosXStd3/results/VOC2012/Segmentation/comp6_trainval_aug_cls';
%save_folder = '/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationClassBboxErodeCRFAug';

orig_folder = '/home/panquwang/Dataset/VOCdevkit/VOC2012/SegmentationClassAug_orig';
save_folder = '/home/panquwang/Dataset/VOCdevkit/VOC2012/SegmentationClassAug';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% You do not need to change values below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
imgs_dir = dir(fullfile(orig_folder, '*.png'));

if ~exist(save_folder, 'dir')
    mkdir(save_folder)
end

for i = 1 : numel(imgs_dir)
    fprintf(1, 'processing %d (%d) ...\n', i, numel(imgs_dir));
    
    img = imread(fullfile(orig_folder, imgs_dir(i).name));
    
    imwrite(img, fullfile(save_folder, imgs_dir(i).name));
end
