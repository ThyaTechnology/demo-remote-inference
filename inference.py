
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from cloudlabeling import cloudlabeling
import cv2
import os
import shutil
import pandas as pd
import copy

parser = ArgumentParser(
    description='Process 1 or more image(s) with a model on CloudLabeling',
    formatter_class=ArgumentDefaultsHelpFormatter)

# Settings
parser.add_argument("--image",  nargs='+', type=str, help="path for 1 or more image(s) to infer")  
parser.add_argument('--verbose', action='store_true', )

# CloudLabeling Settings
parser.add_argument("--project_ID", type=str, default="MSCOCO", help="ID of the project for inference")                   
parser.add_argument("--API_token", type=str, default="", help="ID of the project for inference")                   

# Export Settings
parser.add_argument("--output", nargs='+', type=str, default=None, help="path of 1 or more image(s) to print results")                   
parser.add_argument('--export_csv', type=str, default=None, help="file to export the results in csv format")   
parser.add_argument('--export_xml', action='store_true', help="save detection in xml format")   

# Image Settings
parser.add_argument("--split_image_H", type=int, default=1, help="number of horizontal splits")
parser.add_argument("--split_image_V", type=int, default=1, help="number of vertical splits")  

parser.add_argument('--remove_empty_pixels', action='store_true', help="keep only the smaller patch. Warning: this will change the aspect ratio")  
parser.add_argument("--overlapping_pixels", type=int, default=0, help="amount of overlap between tiles (in pixels)")  

args = parser.parse_args()



cloud_labeler = cloudlabeling.CloudLabeling(api_token=args.API_token)

# loop over list of images
for i_image, image_path in enumerate(args.image):

    # inference on full image
    if args.split_image_H ==1 and args.split_image_V ==1:
        # process full image
        image_results = cloud_labeler.infer_remotely(image_path, project_id=str(args.project_ID), post="curl")

        # check if issue in processing image
        if image_results["error"] is not None:
            print("Error in inference:", patch_results["error"])
    
    # inference on patches
    else:
                
        img = cv2.imread(image_path)

        # Dimensions of the image
        sizeX = img.shape[1]
        sizeY = img.shape[0]

        # number of patch per row and column
        nRows = args.split_image_V
        mCols = args.split_image_H
        

        image_results = {"detection": [], "labels":[]}

        # process patches of images
        for i in range(0, nRows):
            for j in range(0, mCols):
                # define region of interest (roi)
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

                # filename, file_extension = os.path.splitext(image_path)
                
                # i_start = int(i*sizeY/nRows)
                # i_end = int((i+1)*sizeY/nRows)
                # j_start = int(j*sizeX/mCols)
                # j_end = int((j+1)*sizeX/mCols)
                # roi = img[i_start:i_end ,j_start:j_end]

                # store image on temporary storage
                os.makedirs("tmp_patches", exist_ok=True)
                patch_path = 'tmp_patches/patch_'+str(i)+str(j)+".jpg"
                cv2.imwrite(patch_path, roi)

                # process patch of an image
                patch_results = cloud_labeler.infer_remotely(patch_path, project_id=str(args.project_ID), post="curl")

                # check if issue in processing image
                if patch_results["error"] is not None:
                    print("Error in inference:", patch_results["error"])
                    continue
        
                if args.remove_empty_pixels:
                    # convert bounding boxes localization form patch to original image
                    for object in patch_results["detection"]:
                        object["box"] = [x + y for x, y in zip(object["box"], [j_start,i_start,j_start,i_start] )]

                # keep track of all detections so far and a list of unique classes of objects
                image_results["detection"] += patch_results["detection"]
                image_results["labels"] = list(set(image_results["labels"] + patch_results["labels"]))
        
        # remove temporary folder with patches of imags
        shutil.rmtree("tmp_patches")



    # display results on image output
    if args.output is not None:
        image_with_BB = cloud_labeler.display_BB(cv2.imread(image_path), image_results)        
        cv2.imwrite(args.output[i_image], image_with_BB)

    # print all detected objects
    if args.verbose:
        for object in image_results["detection"]:
            print(object)

    # print all classes of objects
    print(f"Classes of objects found in image {image_path}: {image_results['labels']}")
        
    # print the count of each object detected 
    for label in image_results["labels"]:
        nb_object = len([object for object in image_results["detection"] if object["label"] == label])
        print(f"Found {nb_object} {label} in the image {image_path}")

    # export the results into a xml file
    if args.export_xml:
        from pascal_voc_writer import Writer

        img = cv2.imread(image_path)

        # Dimensions of the image
        sizeX = img.shape[1]
        sizeY = img.shape[0]
        # create pascal voc writer (image_path, width, height)
        writer = Writer(image_path, sizeX, sizeY)

        # add objects (class, xmin, ymin, xmax, ymax)
        for object in image_results["detection"]:
            writer.addObject(object["label"], 
                int(object["box"][0]), int(object["box"][1]), 
                int(object["box"][2]), int(object["box"][3]))

        # write to file
        writer.save(image_path[:-4] + '.xml')


    # export the results into a csv file
    if args.export_csv is not None:
        data = {"image": image_path}
        for label in image_results["labels"]:
            data[label] = [len([object for object in image_results["detection"] if object["label"] == label])]

        columns = image_results["labels"]
        columns.insert(0, "image")
        df_img = pd.DataFrame(data, columns = columns)

        if os.path.exists(args.export_csv):
            df_out = pd.read_csv(args.export_csv)  
            df_img = pd.concat([df_out, df_img])

        df_img.to_csv(args.export_csv, mode='w',index=False)

