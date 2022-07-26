# Remote inference

## Commande line inference

```
curl -H "Content-Type: image/jpeg" -H "project_id: XXX" -H "api_token: XXXXX" -X POST --data-binary @sample.jpg http://cloudlabeling.org:4000/api/predict
```

Arguments:
 - project_id: ID of the project to infer from
 - api_token: user identification token

## Create the environment
```
conda create -n cloudlabeling-remoteinference python=3.9
conda activate cloudlabeling-remoteinference
pip install cloudlabeling pandas opencv-python
```


## Run on a single image

```
python inference.py --project_ID XXX --API_token XXXXX --image sample.jpg
```

Arguments:
 - project_id: ID of the project to infer from
 - api_token: user identification token
 - image: local path of the image to process

## Run on more than 1 image

```
python inference.py --project_ID XXX --API_token XXXXX --image sample.jpg sample2.jpg
python inference.py --project_ID XXX --API_token XXXXX --image sample*.jpg
```

Arguments:
 - project_id: ID of the project to infer from
 - api_token: user identification token
 - image: local path of one or more image(s) to process

## Save results on csv

```
python inference.py --project_ID XXX --API_token XXXXX --image sample*.jpg --export_csv results.csv
```

## Run on split image(s)

```
python inference.py --project_ID XXX --API_token XXXXX --split_image_H 2 --split_image_V 2 --image sample.jpg
python inference.py --project_ID XXX --API_token XXXXX --split_image_H 2 --split_image_V 2 --image sample*.jpg
```

Arguments:
 - project_id: ID of the project to infer from
 - api_token: user identification token
 - image: local path of one or more image(s) to process
 - split_image_V: number of vertical split for the image
 - split_image_H: number of horizontal split for the image
 - remove_empty_pixels: remove empty pixels and create smaller-size images
 - overlapping_pixels: number of overlapping pixel on each patch

## Extra

### Split image(s) into patches

```
python split_image.py --split_image_H 2 --split_image_V 2 --output_folder patches --image sample.jpg 
python split_image.py --split_image_H 2 --split_image_V 2 --output_folder patches --image sample*.jpg 
python split_image.py --split_image_H 2 --split_image_V 2 --output_folder patches --input_folder folder_full_of_images/
```

Arguments:
 - split_image_V: number of vertical split for the image
 - split_image_H: number of horizontal split for the image
 - output_folder: local path of a folder to store the split images
 - image: local path of one or more image(s) to process
 - input_folder: local path of a folder with images to process
 - remove_empty_pixels: remove empty pixels and create smaller-size images
 - overlapping_pixels: number of overlapping pixel on each patch