#%%
from ultralytics import YOLO
# %% Load a model
model = YOLO("yolov8n.pt")  # load our custom trained model

# %%
result = model("/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/PyTorch/PyTorch-course/060_CNN_ImageClassification/kiki.jpg")
# %%
result
# %% command line run
# Standard Yolo
!yolo detect predict model=yolov8n.pt source="/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/PyTorch/PyTorch-course/060_CNN_ImageClassification/kiki.jpg" conf=0.3 
#%% Masks without training (detects person and wrongly detects mask)
!yolo detect predict model=yolov8n.pt source="/Users/gastoncrecikeinbaum/Documents/maskvideoyolo.mov" conf=0.3 
# %% Masks 
!yolo detect predict model="/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/PyTorch/PyTorch-course/075_ObjectDetection_Yolo7/runs/detect/train10/weights/best.pt" source="/Users/gastoncrecikeinbaum/Documents/maskvideoyolo.mov" conf=0.3 

# %%
