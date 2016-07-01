function []=SemanticSegmentationEvaluationDeepLabV2()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pascal VOC 2012 Semantic Segmentation Class Result based on DeepLabV2_VGG
%
%
% Note: Before running this code, make sure your output png/mat file is
% located at
% ${VOCdevkit_location}/results/VOC2012/Segmentation/${id}_${VOCopts.testset}_cls/,
% where id is your model name. For example, DeepLabV2_VGG.
% 
% Note VOCopts.testset is either val or test.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% change this path if you installed the VOC elsewhere
VOCdevkit_location='/home/panquwang/Dataset/VOCdevkit/';
addpath([VOCdevkit_location 'VOCcode']);
% the generated default colormap
load pascal_seg_colormap.mat 

% % Define the id of your model. The 
% id='DeepLabV2_VGG16'; % 85.676%
% id='DeepLabV2_VGG16_CRF'; % result: 87.948%. Test 72.42830%
% id='DeepLabV2_VGG16_CRF_png'; % result: 87.948%
% id='DeepLabV2_VGG16_CRF_panqu'; % result: 87.948%
% id='DeepLabV2_resnet101'; % result: 90.912%
id='DeepLabV2_resnet101_CRF'; % result: 91.669%. Test 79.73580%

% initialize VOC options
VOCinit;

%% Initialization
% In general we use mat file so is_png=0; set to 1 if you want to evaluate from png files 
is_png=0;
% set to 1 only when we want to convert mat to png 
convert_to_png=1;
% if we use test set instead of validation set, is_test=1. Note we do
% not have ground truth for test set, we we use original image as "ground truth"
% for code consistency.
is_test=1;
if is_test
    VOCopts.testset='test';
    % original image location
    VOCopts.seg.clsimgpath=[VOCdevkit_location 'VOC2012/JPEGImages/%s.jpg'];    
    % output result png image location
    save_result_folder=[VOCdevkit_location 'results/VOC2012/Segmentation/' id '_' VOCopts.testset '_mat_to_png_cls'];
    mkdir(save_result_folder);
end
% Here we still want to convert to png even if using validation images.
if convert_to_png
    save_result_folder=[VOCdevkit_location 'results/VOC2012/Segmentation/' id '_' VOCopts.testset '_mat_to_png_cls'];
    mkdir(save_result_folder);
end


%% Evaluation
% image val/test set
[gtids,t]=textread(sprintf(VOCopts.seg.imgsetpath,VOCopts.testset),'%s %d');

% number of labels = number of classes plus one for the background
num = VOCopts.nclasses+1; 
confcounts = zeros(num);
count=0;
tic;
for i=1:length(gtids)
    % display progress
    if toc>1
        fprintf('test confusion: %d/%d\n',i,length(gtids));
        drawnow;
        tic;
    end
        
    imname = gtids{i};
    
    
    % ground truth label file
    gtfile = sprintf(VOCopts.seg.clsimgpath,imname);
    [gtim,map] = imread(gtfile);    
    gtim = double(gtim);
    
    
    if is_png == 1
        % results file, if use png file
        resfile = sprintf(VOCopts.seg.clsrespath,id,VOCopts.testset,imname);
        [resim,map] = imread(resfile);
        resim = double(resim);
    else
        % results file, if use mat file. NOTE WE USE .MAT HERE!
        VOCopts.seg.clsrespathmat=[VOCdevkit_location 'results/VOC2012/Segmentation/%s_%s_cls/%s.mat'];
        resfile = sprintf(VOCopts.seg.clsrespathmat,id,VOCopts.testset,[imname '_blob_0']);
        load(resfile);
%         resim = double(imrotate(flip(data),270)); % this line and the
%         line below is equal
        resim=permute(data,[2,1,3]);
        resim=resim(1:size(gtim,1),1:size(gtim,2));
        resim_saved=uint8(resim);
        if convert_to_png==1            
            imwrite(resim_saved, cmap, fullfile(save_result_folder, [imname, '.png']));      
            if is_test == 1
                continue;
            end
        end
        
        
    end
    
    
    

    
    
    
    % Check validity of results image
    maxlabel = max(resim(:));
    if (maxlabel>VOCopts.nclasses), 
        error('Results image ''%s'' has out of range value %d (the value should be <= %d)',imname,maxlabel,VOCopts.nclasses);
    end

    szgtim = size(gtim); szresim = size(resim);
    if any(szgtim~=szresim)
        error('Results image ''%s'' is the wrong size, was %d x %d, should be %d x %d.',imname,szresim(1),szresim(2),szgtim(1),szgtim(2));
    end
    
    %pixel locations to include in computation, which exclude the boundary.
    locs = gtim<255;
    
    % joint histogram, enlarge the values in resim by 21 times (to enable the next hisogram step)
    sumim = 1+gtim+resim*num; 
    hs = histc(sumim(locs),1:num*num); 
    count = count + numel(find(locs)); % total pixels to be calculated
    confcounts(:) = confcounts(:) + hs(:);
end

% confusion matrix - first index is true label, second is inferred label
%conf = zeros(num);
conf = 100*confcounts./repmat(1E-20+sum(confcounts,2),[1 size(confcounts,2)]);
rawcounts = confcounts;

% Percentage correct labels measure is no longer being used.  Uncomment if
% you wish to see it anyway
%overall_acc = 100*sum(diag(confcounts)) / sum(confcounts(:));
%fprintf('Percentage of pixels correctly labelled overall: %6.3f%%\n',overall_acc);

accuracies = zeros(VOCopts.nclasses,1);
fprintf('Accuracy for each class (intersection/union measure)\n');
for j=1:num
   
   gtj=sum(confcounts(j,:));
   resj=sum(confcounts(:,j));
   gtjresj=confcounts(j,j);
   % The accuracy is: true positive / (true positive + false positive + false negative) 
   % which is equivalent to the following percentage:
   accuracies(j)=100*gtjresj/(gtj+resj-gtjresj);   
   
   clname = 'background';
   if (j>1), clname = VOCopts.classes{j-1};end;
   fprintf('  %14s: %6.3f%%\n',clname,accuracies(j));
end
accuracies = accuracies(1:end);
avacc = mean(accuracies);
fprintf('-------------------------\n');
fprintf('Average accuracy: %6.3f%%\n',avacc);
