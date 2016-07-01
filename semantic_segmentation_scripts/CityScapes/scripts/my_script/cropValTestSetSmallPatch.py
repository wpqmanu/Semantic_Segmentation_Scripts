#!/usr/bin/python
#
# Making val/test files that contains smaller image patches suitable for caffe testing.
#
# The raw image path of CityScapes datasets is /home/panquwang/Dataset/CityScapes/leftImg8bit_trainvaltest/leftImg8bit/(train,val,test)
# The ground truth path for Fine images is /home/panquwang/Dataset/CityScapes/gtFine/(train,val,test)

# python imports
from __future__ import print_function
import os, glob, sys
import cv2
import numpy as np

# cityscapes imports
sys.path.append( os.path.normpath( os.path.join( os.path.dirname( __file__ ) , '..' , 'helpers' ) ) )

# The main method
def main(selected_set,crop_size_h,crop_size_w):
    # Where to look for Cityscapes
    if 'CITYSCAPES_DATASET' in os.environ: # cityscapesPath=/home/panquwang/Dataset/CityScapes
        cityscapesPath = os.environ['CITYSCAPES_DATASET']
    else:
        print ("Make sure you have added CITYSCAPES_DATASET in your python path!")
        exit()


    #TODO:HERE. FOR BOTH ORIGINAL IMAGES AND GROUND TRUTH.
    gt_type = 'fine'

    # how to search for the original images
    search_original = os.path.join(cityscapesPath, "leftImg8bit_trainvaltest", "leftImg8bit", selected_set, "*","*leftImg8bit.png")

    # how to search for all ground truth
    search_gt = {}
    search_gt[gt_type] = os.path.join(cityscapesPath, "gtFine", selected_set, "*", "*_labelTrainIds.png")
    files_gt = {}
    files_gt[gt_type] = glob.glob(search_gt[gt_type])
    files_gt[gt_type].sort()


    # search files
    files_original = glob.glob(search_original)
    files_original.sort()


    # Making new directories for patch files
    new_original_image_directory=os.path.join(cityscapesPath, "leftImg8bit_trainvaltest", "leftImg8bit", selected_set+"_patches")
    new_gt_directory = os.path.join(cityscapesPath, "gtFine", selected_set+"_patches")
    if not os.path.exists(new_original_image_directory) and not os.path.exists(new_gt_directory):
        os.makedirs(new_original_image_directory)
        os.makedirs(new_gt_directory)

    print ("Start cropping files...")
    for image_path,gt_path in zip(files_original,files_gt[gt_type]):
        image_original=cv2.imread(image_path,1)
        image_gt=cv2.imread(gt_path,0)
        height,width=image_original.shape[0],image_original.shape[1]
        total_patches=0
        # Note we knwow the size of the images are 1024x2048, so no padding is needed
        for i in range(height/crop_size_h):
            for j in range(width/crop_size_w):
                image_original_temp=image_original[i*crop_size_h:(i+1)*crop_size_h,j*crop_size_w:(j+1)*crop_size_w,:]
                image_gt_temp = image_gt[i * crop_size_h:(i + 1) * crop_size_h, j * crop_size_w:(j + 1) * crop_size_w]

                original_file_names_split = image_path.split("/")
                original_file_patch_name = original_file_names_split[-1][:-4]+'_patch'+str(total_patches)+'.png'
                original_file_patch_saved_location=new_original_image_directory+'/'+original_file_patch_name
                cv2.imwrite(original_file_patch_saved_location,image_original_temp)


                gt_file_names_split = gt_path.split("/")
                gt_file_patch_name = gt_file_names_split[-1][:-4]+'_patch'+str(total_patches)+'.png'
                gt_file_patch_saved_location=new_gt_directory+'/'+gt_file_patch_name
                cv2.imwrite(gt_file_patch_saved_location, image_gt_temp)


                total_patches=total_patches+1
                print("Finished making {}...".format(original_file_patch_name))

    print("Finished.")


# call the main
if __name__ == "__main__":
    set_to_be_processed=['test']
    crop_size_h=512
    crop_size_w=512
    for selected_set in set_to_be_processed:
        main(selected_set,crop_size_h,crop_size_w)
