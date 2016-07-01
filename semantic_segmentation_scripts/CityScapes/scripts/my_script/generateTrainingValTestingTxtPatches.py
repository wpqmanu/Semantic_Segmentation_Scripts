#!/usr/bin/python
#
# Making val_patch/test_patch.txt files that are suitable for caffe validation/testing.
#
# The raw image patch path of CityScapes datasets is /home/panquwang/Dataset/CityScapes/leftImg8bit_trainvaltest/leftImg8bit/(val_patches,test_patches)
# The ground truth path for Fine images is /home/panquwang/Dataset/CityScapes/gtFine/(val_patches)

# python imports
from __future__ import print_function
import os, glob, sys

# cityscapes imports
sys.path.append( os.path.normpath( os.path.join( os.path.dirname( __file__ ) , '..' , 'helpers' ) ) )

# The main method
def main(selected_set):
    # Where to look for Cityscapes
    if 'CITYSCAPES_DATASET' in os.environ: # cityscapesPath=/home/panquwang/Dataset/CityScapes
        cityscapesPath = os.environ['CITYSCAPES_DATASET']
    else:
        print ("Make sure you have added CITYSCAPES_DATASET in your python path!")
        exit()

    gt_types = ['fine']

    # how to search for the original image patches
    search_original=os.path.join( cityscapesPath , "leftImg8bit_trainvaltest"   , "leftImg8bit" , selected_set , "*.png" )

    # how to search for all ground truth image patches
    search_gt={}
    search_gt['fine']   = os.path.join( cityscapesPath , "gtFine" , selected_set , "*.png" )

    # search files
    filesOriginal=glob.glob(search_original)
    filesOriginal.sort()

    # Making *.txt files
    created_list_location = cityscapesPath + '/created_list/'
    if not os.path.exists(created_list_location):
        os.mkdir(created_list_location)

    for gt_type in gt_types:
        filesGT={}
        filesGT[gt_type] = glob.glob( search_gt[gt_type] )
        filesGT[gt_type].sort()

        # quit if we did not find anything
        if not filesOriginal or not filesGT:
            print( "Did not find any files. Please check your path for Original images, Fine and Coarse label" )
            exit()

        print("Making {}_{}_patches.txt files...".format(selected_set,gt_type))
        gt_content=filesGT[gt_type]
        file_name=created_list_location+'{}_{}.txt'.format(selected_set,gt_type)
        file_name_id = created_list_location + '{}_{}_id.txt'.format(selected_set, gt_type)

        if not os.path.isfile(file_name):
            with open(file_name, 'w') as f:
                if selected_set != 'test' and selected_set != 'test_patches':
                    for current_row_id,current_row_content in enumerate(filesOriginal):
                        f.write(current_row_content+' '+gt_content[current_row_id]+'\n')
                else:
                    for current_row_id, current_row_content in enumerate(filesOriginal):
                        f.write(current_row_content+'\n')

        print("Finished making {}_{}.txt files...".format(selected_set,gt_type))

        if not os.path.isfile(file_name_id):
            with open(file_name_id, 'w') as f_id:
                for current_row_id_id, current_row_content_id in enumerate(filesOriginal):
                    current_row_content_split=current_row_content_id.split("/")
                    current_id=current_row_content_split[-1][:-4]
                    f_id.write(current_id + '\n')

        print("Finished making {}_{}_id.txt files...".format(selected_set, gt_type))


# call the main
if __name__ == "__main__":
    # set_to_be_processed=['train_patches','val_patches','test_patches']
    set_to_be_processed = ['val_big_patches']
    for selected_set in set_to_be_processed:
        main(selected_set)
