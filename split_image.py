
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import cv2
import os
import shutil


parser = ArgumentParser(
    description='Split 1 or more image(s) into parts',
    formatter_class=ArgumentDefaultsHelpFormatter)

# Settings
parser.add_argument("--image",  nargs='+', type=str, help="path for 1 or more image(s) to infer")  
parser.add_argument("--input_folder", type=str, default=None, help="path for 1 or more image(s) to infer")  
parser.add_argument('--verbose', action='store_true', )
parser.add_argument("--output_folder", type=str, default="patches", help="folder path to sotre the patches")                   

# Image Settings
parser.add_argument("--split_image_H", type=int, default=2, help="number of horizontal splits")
parser.add_argument("--split_image_V", type=int, default=2, help="number of vertical splits")  

parser.add_argument('--remove_empty_pixels', action='store_true', help="keep only the smaller patch. Warning: this will change the aspect ratio")  
parser.add_argument("--overlapping_pixels", type=int, default=0, help="amount of overlap between tiles (in pixels)")  

args = parser.parse_args()


# if a folder is provided, find all images in this folder
if args.input_folder is not None:
    supported_extensions = ('.jpg', '.jpeg', '.png', '.tiff', '.tif')
    args.image = [os.path.join(args.input_folder, filename) for filename in os.listdir(args.input_folder) if filename.lower().endswith(supported_extensions)]
    # print(args.image)

# loop over list of images
for i_image, image_path in enumerate(args.image):

    # print(image_path)
    img = cv2.imread(image_path)

    # Dimensions of the image
    sizeX = img.shape[1]
    sizeY = img.shape[0]

    # number of patch per row and column
    nRows = args.split_image_V
    mCols = args.split_image_H
    

    image_results = {"detection": [], "labels":[]}
    import copy
    # process patches of images
    for i in range(0, nRows):
        for j in range(0, mCols):
            i_start = max(int(i*sizeY/nRows) - args.overlapping_pixels, 0)
            i_end = min(int((i+1)*sizeY/nRows) + args.overlapping_pixels, sizeY)
            j_start = max(int(j*sizeX/mCols) - args.overlapping_pixels, 0)
            j_end = min(int((j+1)*sizeX/mCols)+ args.overlapping_pixels, sizeX)
                
            if args.remove_empty_pixels:
                roi = img[i_start:i_end ,j_start:j_end]
            else:
                roi = copy.deepcopy(img)
                roi[:i_start ,:] = 0
                roi[i_end: ,:] = 0
                roi[: ,:j_start] = 0
                roi[: ,j_end:] = 0 

            filename, file_extension = os.path.splitext(image_path)

            # print(f"{filename}_{i}{j}{file_extension}")
            patch_path = os.path.join(args.output_folder, f"{os.path.basename(filename)}_{str(i)}{str(j)}{file_extension}")
            os.makedirs(os.path.dirname(patch_path), exist_ok=True)
            cv2.imwrite(patch_path, roi)

    
