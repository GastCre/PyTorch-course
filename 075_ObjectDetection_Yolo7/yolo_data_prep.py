#%% package import
import pandas as pd
import numpy as np
import seaborn as sns
import os
import shutil
import xml.etree.ElementTree as ET
import glob

import json
#%% Function for conversion XML to YOLO
# based on https://towardsdatascience.com/convert-pascal-voc-xml-to-yolo-for-object-detection-f969811ccba5
def xml_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]

#%% create folders
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Use forward slashes for macOS/Unix compatibility
create_folder('/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/PyTorch/PyTorch-course/075_ObjectDetection_Yolo7/yolov7/train/images')
create_folder('/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/PyTorch/PyTorch-course/075_ObjectDetection_Yolo7/yolov7/train/labels')
create_folder('/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/PyTorch/PyTorch-course/075_ObjectDetection_Yolo7/yolov7/val/images')
create_folder('/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/PyTorch/PyTorch-course/075_ObjectDetection_Yolo7/yolov7/val/labels')
create_folder('/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/PyTorch/PyTorch-course/075_ObjectDetection_Yolo7/yolov7/test/images')
create_folder('/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/PyTorch/PyTorch-course/075_ObjectDetection_Yolo7/yolov7/test/labels')

#%% First, discover what classes actually exist in your XML files
annotations_folder = '/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/PyTorch/PyTorch-course/075_ObjectDetection_Yolo7/annotations'
classes_found = set()

print("Scanning XML files to discover classes...")
for xml_file in os.listdir(annotations_folder):
    if xml_file.endswith('.xml'):
        try:
            tree = ET.parse(os.path.join(annotations_folder, xml_file))
            root = tree.getroot()
            for obj in root.findall('object'):
                label = obj.find("name").text
                classes_found.add(label)
        except Exception as e:
            print(f"Error reading {xml_file}: {e}")

print(f"Classes found in XML files: {sorted(classes_found)}")

# Define expected classes
expected_classes = ['with_mask', 'without_mask', 'mask_weared_incorrect']
classes = expected_classes

# Check if all expected classes are found
missing_classes = set(expected_classes) - classes_found
extra_classes = classes_found - set(expected_classes)

if missing_classes:
    print(f"WARNING: Expected classes not found in XML: {missing_classes}")
if extra_classes:
    print(f"WARNING: Unexpected classes found in XML: {extra_classes}")

print(f"Using classes: {classes}")

#%% get all image files
img_folder = '/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/PyTorch/PyTorch-course/075_ObjectDetection_Yolo7/images'
_, _, files = next(os.walk(img_folder))

# Filter for image files only
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
files = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]

print(f"Found {len(files)} image files")

pos = 0
processed_images = 0
processed_labels = 0
skipped_labels = 0
class_counts = {i: 0 for i in range(len(classes))}

for f in files:
    source_img = os.path.join(img_folder, f)
    
    # Determine destination folder based on position
    if pos < 700:
        dest_folder = '/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/PyTorch/PyTorch-course/075_ObjectDetection_Yolo7/yolov7/train'
        split_name = "train"
    elif (pos >= 700 and pos < 800):
        dest_folder = '/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/PyTorch/PyTorch-course/075_ObjectDetection_Yolo7/yolov7/val'
        split_name = "val"
    else:
        dest_folder = '/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/PyTorch/PyTorch-course/075_ObjectDetection_Yolo7/yolov7/test'
        split_name = "test"
    
    destination_img = os.path.join(dest_folder, 'images', f)
    shutil.copy(source_img, destination_img)
    processed_images += 1

    # check for corresponding label
    label_file_basename = os.path.splitext(f)[0]
    label_source_file = f"{label_file_basename}.xml"
    label_dest_file = f"{label_file_basename}.txt"
    
    label_source_path = os.path.join(annotations_folder, label_source_file)
    label_dest_path = os.path.join(dest_folder, 'labels', label_dest_file)
    
    # if file exists, copy it to target folder
    if os.path.exists(label_source_path):
        try:
            # parse the content of the xml file
            tree = ET.parse(label_source_path)
            root = tree.getroot()
            width = int(root.find("size").find("width").text)
            height = int(root.find("size").find("height").text)
            
            result = []
            objects_in_image = 0
            
            for obj in root.findall('object'):
                label = obj.find("name").text
                
                # Check if label is in our expected classes
                if label in classes:
                    index = classes.index(label)
                    class_counts[index] += 1
                    
                    pil_bbox = [int(x.text) for x in obj.find("bndbox")]
                    yolo_bbox = xml_to_yolo_bbox(pil_bbox, width, height)
                    
                    # Validate bbox coordinates (should be between 0 and 1)
                    if all(0 <= coord <= 1 for coord in yolo_bbox):
                        bbox_string = " ".join([str(x) for x in yolo_bbox])
                        result.append(f"{index} {bbox_string}")
                        objects_in_image += 1
                    else:
                        print(f"Invalid bbox coordinates in {f}: {yolo_bbox}")
                else:
                    print(f"Unknown class '{label}' in {f}, skipping object")
            
            if result:
                # generate a YOLO format text file for each xml file
                with open(label_dest_path, "w", encoding="utf-8") as file_out:
                    file_out.write("\n".join(result))
                processed_labels += 1
                print(f"Processed {f} -> {split_name} ({objects_in_image} objects)")
            else:
                # Create empty label file if no valid objects found
                with open(label_dest_path, "w", encoding="utf-8") as file_out:
                    file_out.write("")
                print(f"No valid objects in {f}, created empty label file")
                
        except Exception as e:
            print(f"Error processing {label_source_path}: {e}")
            skipped_labels += 1
    else:
        print(f"No annotation file found for {f}")
        # Create empty label file for images without annotations
        with open(label_dest_path, "w", encoding="utf-8") as file_out:
            file_out.write("")
        skipped_labels += 1
    
    pos += 1

# Print summary
print("\n" + "="*50)
print("DATA PREPARATION SUMMARY")
print("="*50)
print(f"Total images processed: {processed_images}")
print(f"Labels with objects: {processed_labels}")
print(f"Labels without objects: {skipped_labels}")
print(f"Classes distribution:")
for i, class_name in enumerate(classes):
    print(f"  {class_name}: {class_counts[i]} objects")
print("="*50)

# %%
