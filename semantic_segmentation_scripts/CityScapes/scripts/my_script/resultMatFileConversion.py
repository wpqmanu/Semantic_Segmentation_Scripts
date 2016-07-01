#!/usr/bin/python
#
# Convert the generated .mat files in DeepLab_V2_ResNet101 to png file.
#
# The raw image patch path of CityScapes datasets is /home/panquwang/Dataset/CityScapes/leftImg8bit_trainvaltest/leftImg8bit/(val_patches,test_patches)
# The ground truth path for Fine images is /home/panquwang/Dataset/CityScapes/gtFine/(val_patches)

# python imports
from __future__ import print_function
import os, glob, sys
import scipy.io as sio
import numpy as np
import cv2
from shutil import copyfile
from matplotlib import pyplot as plt



# cityscapes imports
sys.path.append( os.path.normpath( os.path.join( os.path.dirname( __file__ ) , '..' , 'helpers' ) ) )
from labels     import trainId2label

def main(selected_set,result_mat_location,crop_size):
    # Where to look for Cityscapes
    if 'CITYSCAPES_DATASET' in os.environ: # cityscapesPath=/home/panquwang/Dataset/CityScapes
        cityscapesPath = os.environ['CITYSCAPES_DATASET']
    else:
        print ("Make sure you have added CITYSCAPES_DATASET in your python path!")
        exit()

    result_mat_location_full=os.path.join(cityscapesPath,"results_saved",result_mat_location)

    gt_types = ['fine']

    # how to search for all .mat patch files
    files_mat=glob.glob(result_mat_location_full+"/*.mat")
    files_mat.sort()

    # Making *.png files -> first we should have a location
    created_list_location = os.path.join(cityscapesPath,"results_saved",result_mat_location+'_png')
    if not os.path.exists(created_list_location):
        os.mkdir(created_list_location)

    final_height=1024
    final_width=2048
    total_mat_patches_per_image=8
    total_images=len(files_mat)/total_mat_patches_per_image


    # list over for all files
    for current_image_index in range(total_images):
        index_start=current_image_index*total_mat_patches_per_image
        index_end=(current_image_index+1)*total_mat_patches_per_image
        final_array=np.empty([final_height,final_width])
        # concatenate all files
        for index,mat_file_name in enumerate(files_mat[index_start:index_end]):
            mat_file_content=sio.loadmat(mat_file_name)['data']
            print("{} loaded".format(mat_file_name))

            mat_file_content=np.squeeze(mat_file_content)
            mat_file_content=np.transpose(mat_file_content)
            # plt.imshow(mat_file_content, interpolation='nearest')
            # plt.show()
            mat_file_content=mat_file_content[:crop_size,:crop_size]
            # plt.imshow(mat_file_content, interpolation='nearest')
            # plt.show()
            if index<4:
                final_array[:final_height/2,crop_size*index:crop_size*(index+1)]=mat_file_content
            else:
                final_array[final_height/2:,crop_size*(index-4):crop_size*(index-4+1)]=mat_file_content


        # Here we perform the conversion from train_id to id
        unique_values_in_final_array=np.unique(final_array)
        unique_values_in_final_array=np.sort(unique_values_in_final_array)
        unique_values_in_final_array=unique_values_in_final_array[::-1]

        for unique_value in unique_values_in_final_array:
            converted_value=trainId2label[unique_value].id
            final_array[final_array == unique_value] = converted_value

        # # display image
        # plt.imshow(final_array, interpolation='nearest')
        # plt.show()

        # save to png images
        mat_name_split=mat_file_name.split("/")
        to_be_saved_png_file_name=created_list_location+'/'+mat_name_split[-1][:-4]+'.png'
        cv2.imwrite(to_be_saved_png_file_name,final_array)

        copyfile(to_be_saved_png_file_name, cityscapesPath+'/results/'+mat_name_split[-1][:-4]+'.png')

# call the main
if __name__ == "__main__":
    set_to_be_processed=['val_patches']
    result_mat_location="DeepLabV2_ResNet101_patches_mat_iteration_4500"
    crop_size=512
    for selected_set in set_to_be_processed:
        main(selected_set,result_mat_location,crop_size)
