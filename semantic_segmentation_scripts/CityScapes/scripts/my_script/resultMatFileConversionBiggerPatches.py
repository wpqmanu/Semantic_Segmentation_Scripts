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

def get_color_map():
    # Custom colormaps
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    lut[:, 0, 2] = [0, 0, 0, 0, 0, 111, 81, 128, 244, 250, 230, 70, 102, 190, 180, 150, 150, 153, 153, 250,
                    220, 107, 152, 70, 220, 255, 0, 0, 0, 0, 0, 0, 0, 119, 0]+([255] * 221)
    lut[:, 0, 1] = [0, 0, 0, 0, 0, 74, 0, 64, 35, 170, 150, 70, 102, 153, 165, 100, 120, 153, 153, 170,
                    220, 142, 251, 130, 20, 0, 0, 0, 60, 0, 0, 80, 0, 11, 0]+([255] * 221)
    lut[:, 0, 0] = [0, 0, 0, 0, 0, 0, 81, 128, 232, 160, 140, 70, 156, 153, 180, 100, 90, 153, 153, 30,
                    0, 35, 152, 180, 60, 0, 142, 70, 100, 90, 110, 100, 230, 32, 142]+([255] * 221)
    return lut

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

    previous_padding_per_side=13
    final_height=1024
    final_width=2048
    total_mat_patches_per_image=8
    total_images=len(files_mat)/total_mat_patches_per_image
    lut=get_color_map()


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
            if index==0:
                mat_file_content = mat_file_content[:crop_size, :crop_size]
            if index == 1 or index ==2:
                mat_file_content = mat_file_content[:crop_size, previous_padding_per_side:previous_padding_per_side+crop_size]
            if index == 3:
                mat_file_content = mat_file_content[:crop_size, 2*previous_padding_per_side:2*previous_padding_per_side + crop_size]
            if index==4:
                mat_file_content = mat_file_content[2*previous_padding_per_side:2*previous_padding_per_side+crop_size, :crop_size]
            if index == 5 or index ==6:
                mat_file_content = mat_file_content[2*previous_padding_per_side:2*previous_padding_per_side+crop_size, previous_padding_per_side:previous_padding_per_side+crop_size]
            if index == 7:
                mat_file_content = mat_file_content[2*previous_padding_per_side:2*previous_padding_per_side+crop_size, 2*previous_padding_per_side:2*previous_padding_per_side + crop_size]

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

        final_array=np.uint8(final_array)
        final_array_color = cv2.cvtColor(final_array, cv2.COLOR_GRAY2RGB)
        final_array_color = cv2.LUT(final_array_color, lut)

        # # display image. Note we have to convert to BGR in matplotlib
        # final_array_color_BGR = cv2.cvtColor(final_array, cv2.COLOR_GRAY2RGB)
        # final_array_color_BGR = cv2.LUT(final_array_color_BGR, lut)
        # plt.imshow(final_array_color_BGR)
        # plt.show()

        # save to png images
        mat_name_split=mat_file_name.split("/")
        to_be_saved_png_file_name=created_list_location+'/'+mat_name_split[-1][:-4]+'.png'
        # color image
        cv2.imwrite(to_be_saved_png_file_name,final_array_color)
        # grayscale label image
        cv2.imwrite(cityscapesPath+'/results/'+mat_name_split[-1][:-4]+'.png', final_array)

        # copyfile(to_be_saved_png_file_name, cityscapesPath+'/results/'+mat_name_split[-1][:-4]+'.png')

# call the main
if __name__ == "__main__":
    set_to_be_processed=['val_patches']
    # result_mat_location="DeepLabV2_ResNet101_patches_mat_iteration_31200_big_patches"
    result_mat_location = "DeepLabV2_ResNet101_patches_mat_iteration_14500_pc_training_big_patches"
    crop_size=512
    for selected_set in set_to_be_processed:
        main(selected_set,result_mat_location,crop_size)
