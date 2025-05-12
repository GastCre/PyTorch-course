
#%%
import sys
sys.path.append('/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/PyTorch/PyTorch-course/065_AudioClassification/')
import torchaudio
from plot_audio import plot_specgram, plot_waveform
import seaborn as sns   
import matplotlib.pyplot as plt
import re
import os

#%%
data_path='/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/PyTorch/PyTorch-course/065_AudioClassification/data/set_a'
data_names=os.listdir(data_path)
#%%
# We apply the name extraction function to all the elements of the list and then extract the unique values
# This works only if there are no folders created yet
tags=list(dict.fromkeys(list(map(lambda x: re.findall(r'(\w+)__',x)[0],data_names))))
print(tags)
#%% Now we iterate over the names, create the spectrogram and save it
# in the corresponding folder
for name in data_names:
    tag, id =zip(*re.findall(r'(\w+)__(\w+).wav',name))
    tag=list(tag)[0]
    id=list(id)[0]
    for class_tag in tags:
        if tag == class_tag:
            if not os.path.isdir(data_path+'/'+class_tag):
                os.makedirs(data_path+'/'+class_tag)
            file_path=data_path+'/'+ name
            data_waveform, sr = torchaudio.load(file_path)
            plot_specgram(waveform=data_waveform, sample_rate=sr,file_path=data_path+'/'+class_tag+'/'+id+'.png')
