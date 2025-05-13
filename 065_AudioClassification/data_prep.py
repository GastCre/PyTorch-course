
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
matching_files = [x for x in data_names if re.findall(r'(\w+)__', x)]
tags = list(dict.fromkeys([re.findall(r'(\w+)__', x)[0] for x in matching_files]))
print(tags)
#%% 
# We have to separate between test and train images, so we define the test size
train_size=0.2
#%%
# Now we iterate over the names, create the spectrogram and save it
# in the corresponding folder

#We count the file number
file_n=0
for name in data_names:
    tag, id =zip(*re.findall(r'(\w+)__(\w+).wav',name))
    tag=list(tag)[0]
    id=list(id)[0]
    for class_tag in tags:
        if tag == class_tag:
            if file_n <= round(train_size*len(data_names)):
                if not os.path.isdir(data_path+'/train/'+class_tag):
                    os.makedirs(data_path+'/train/'+class_tag)
                file_path=data_path+'/'+ name
                data_waveform, sr = torchaudio.load(file_path)
                plot_specgram(waveform=data_waveform, sample_rate=sr,file_path=data_path+'/train/'+class_tag+'/'+id+'.png')
                file_n=file_n+1
            else:
                if not os.path.isdir(data_path+'/test/'+class_tag):
                    os.makedirs(data_path+'/test/'+class_tag)
                file_path=data_path+'/'+ name
                data_waveform, sr = torchaudio.load(file_path)
                plot_specgram(waveform=data_waveform, sample_rate=sr,file_path=data_path+'/test/'+class_tag+'/'+id+'.png')
                file_n=file_n+1
#%%
print(round(test_size*len(data_names)))
# %%
