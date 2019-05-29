'''
prepare the training set file list
https://github.com/JiahuiYu/generative_inpainting/issues/15

The directory tree should be looked like:

- model_logs
- neuralgym_logs
- training_data
  -- training
    --- <folder1>
    --- <folder2>
    --- .....
  -- validation
    --- <val_folder1>
    --- <val_folder2>
    --- .....
- flistGenerate.py

'''
import argparse
import os
from random import shuffle
import matplotlib.pyplot as plt
import numpy as np
import skimage
import os
import sys
from skimage import io
import shutil
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument("--command",
                    metavar="<command>", default='image',
                    help="'image' or 'mask'")
parser.add_argument('--rawImg_path', default=r'L:\SharedData\Houston\AAET\DEPT_DATA\Interns_2019\02_Seismic_Compressive_Sensing_Tech\Project\Data\Mozambique', type=str,
                    help='The rawImg path, image could be arbitary size')
parser.add_argument('--write_dir', default=r'dataset\training_data', type=str,
                    help='The folder path')
parser.add_argument('--train_filename', default=r'data_flist\train_shuffled.flist', type=str,
                    help='The train filename.')
parser.add_argument('--validation_filename', default=r'data_flist\validation_shuffled.flist', type=str,
                    help='The validation filename.')
parser.add_argument('--is_shuffled', default='1', type=int,
                    help='Needed to be shuffled')

if __name__ == "__main__":

    args = parser.parse_args()

    if args.command == 'mask':
        # create vertical masks
        mask_opt_ls = ["uniform", "non_uniform"]
        mask_opt = "uniform"
        crop_size = [256, 256]

        ''' training mask  '''
        training_dir = os.path.join( args.write_dir , "training",mask_opt)  # r"C:\Users\ujz689\RebeccaLI\Proj\generative_inpainting\dataset\cropped_mz"
        if not os.path.exists(training_dir):
            os.makedirs(training_dir)
        mask_write_dir = training_dir
        frame_diff = 1    # sample unit
        for grid_width in range (2,10) :
            fill = np.zeros(crop_size[1], dtype=np.uint8) + 1
            for i in range ( 0, crop_size[0], grid_width ):
                fill[i:i+frame_diff] = 0   # sample rate
            mask = np.tile (fill,(crop_size[0] ,1))
            mask_out = np.dstack([mask,mask,mask])*255
            io.imsave (os.path.join(mask_write_dir,"mask_gw" + str(grid_width) + ".png"),mask_out)
        # plt.figure()
        #     plt.subplot(2,4,grid_width-1)
        #     plt.imshow(mask_out)
        #     plt.title(  "grid_width: " + str( grid_width ) + "\ncs_rate:" + str(  1 - fill.sum() /crop_size[1] ))
        # plt.savefig("grid_width to cs_rate.tif")

        ''' validation mask  '''
        validation_dir = os.path.join(args.write_dir, "validation",mask_opt )
        if not os.path.exists(validation_dir):
            os.makedirs(validation_dir)
        mask_write_dir = validation_dir
        frame_diff = 1    # sample unit
        for grid_width in range (2,10) :
            for i in range ( 0, crop_size[0], grid_width ):
                fill = np.zeros(crop_size[1], dtype=np.uint8) + 1
                fill[i:i+frame_diff] = 0   # sample rate
                mask = np.tile (fill,(crop_size[0] ,1))
                mask_out = np.dstack([mask,mask,mask])*255
                io.imsave (os.path.join(mask_write_dir,"mask_gw" + str(grid_width) + "_i" + str (i) + ".png"),mask_out)
    else:
        training_dir = os.path.join( args.write_dir , "training")  # r"C:\Users\ujz689\RebeccaLI\Proj\generative_inpainting\dataset\cropped_mz"
        validation_dir = os.path.join( args.write_dir, "validation")
        if not os.path.exists(training_dir):
            os.makedirs(training_dir)
        if not os.path.exists(validation_dir):
            os.makedirs(validation_dir)

        # Crop image to fix size
        interval = 64  # cropped_mz_full: 1 , cropped_mz_mini: 64
        crop_size = [256, 256]

        for subset in os.listdir (args.rawImg_path):   # xline and inline
            print ("subset = ", subset)
            data_dir = os.path.join (args.rawImg_path ,subset )
            training_dir_subset = os.path.join(training_dir,subset)   # write/temp/xline
            if not os.path.exists(training_dir_subset):
                os.makedirs(training_dir_subset)
            for raw_fName in os.listdir(data_dir):
            # for raw_fName in os.listdir(data_dir)[0:2]:    # debug
                img = io.imread(os.path.join(data_dir, raw_fName))
                for i in range(0, img.shape[0] - crop_size[0], interval):
                    for j in range(0, img.shape[1] - crop_size[1], interval):
                        crop = img[i: i + crop_size[0], j: j + crop_size[1], :-1]  # 256*256*3
                        crop_rv = 255 - crop  # 255 :empty px -> 0: empty
                        if crop_rv[:, 0, :].mean() > 5 and crop_rv[:, -1, :].mean() > 5:  # guarantee the non empty values
                            fName = raw_fName.split(".")[0] + "_" + str(i) + "_" + str(j)
                            print("training Imgid : ", fName)
                            io.imsave(os.path.join(training_dir_subset,
                                                   '{}.{}'.format(fName, raw_fName.split(".")[1])),
                                      crop)
            # separate validation and training ids

            fns_datasets = os.listdir(training_dir_subset)
            shuffle(fns_datasets)
            VAL_IMAGE_IDS  = fns_datasets[0:int(len(fns_datasets)*0.05)]   # validation rate 5%

            validation_dir_subset = os.path.join(validation_dir,subset)   # write/temp/xline
            if not os.path.exists(validation_dir_subset):
                os.makedirs(validation_dir_subset)
            for img_id in fns_datasets:
                if img_id in VAL_IMAGE_IDS:
                    print ("Validation img_id:" , img_id )
                    shutil.move(os.path.join(training_dir_subset ,img_id ),
                                os.path.join(validation_dir_subset ,img_id ))

        # get the list of directories and separate them into 2 types: training and validation
        training_dirs = os.listdir( os.path.join (args.write_dir , "training"))
        validation_dirs = os.listdir( os.path.join ( args.write_dir , "validation"))

        # make 2 lists to save file paths
        training_file_names = []
        validation_file_names = []

        # append all files into 2 lists
        for training_dir in training_dirs:
            # append each file into the list file names
            training_folder = os.listdir( os.path.join ( args.write_dir , "training" ,training_dir))
            for training_item in training_folder:
                # modify to full path -> directory
                training_item = os.path.join ( args.write_dir , "training" , training_dir ,training_item)
                training_file_names.append(training_item)

        # append all files into 2 lists
        for validation_dir in validation_dirs:
            # append each file into the list file names
            validation_folder = os.listdir(os.path.join ( args.write_dir ,"validation" , validation_dir))
            for validation_item in validation_folder:
                # modify to full path -> directory
                validation_item = os.path.join ( args.write_dir ,"validation" , validation_dir , validation_item)
                validation_file_names.append(validation_item)

        # # print all file paths
        # for i in training_file_names:
        #     print(i)
        # for i in validation_file_names:
        #     print(i)


        # shuffle file names if set
        if args.is_shuffled == 1:
            shuffle(training_file_names)
            shuffle(validation_file_names)

        # make output file if not existed
        if not os.path.exists(os.path.split(args.train_filename)[0]):
            os.makedirs(args.train_filename)
        if not os.path.exists(os.path.split(args.validation_filename)[0]):
            os.makedirs(args.validation_filename)

        print ( "...write to file ")
        fo = open(args.train_filename, "w")
        fo.write("\n".join(training_file_names))
        fo.close()

        fo = open(args.validation_filename, "w")
        fo.write("\n".join(validation_file_names))
        fo.close()

        # print process
        print("Written file is: ", args.train_filename, ", is_shuffle: ", args.is_shuffled)