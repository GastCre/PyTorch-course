import os
import shutil
import xml.etree.ElementTree as ET

def xml_to_yolo_bbox(bbox, w, h):
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]

# Check what classes exist in your data first
classes_found = set()
annotations_folder = '/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/PyTorch/PyTorch-course/075_ObjectDetection_Yolo7/annotations'

for xml_file in os.listdir(annotations_folder):
    if xml_file.endswith('.xml'):
        tree = ET.parse(os.path.join(annotations_folder, xml_file))
        root = tree.getroot()
        for obj in root.findall('object'):
            label = obj.find("name").text
            classes_found.add(label)

print("Classes found in XML files:", classes_found)

# Now use these exact class names
classes = list(classes_found)
print("Using classes:", classes)

# Continue with data preparation using the correct class names...