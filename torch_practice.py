from PIL import Image
import matplotlib.pyplot as plt 
from matplotlib.pyplot import imshow
import pandas as pd
import os
import glob
import requests  
import torch
from torch.utils.data import Dataset

def show_data(data_sample,shape=(28,28)):
    plt.imshow(data_sample[0].numpy().reshape(shape),cmap='gray')
    plt.title('y = '+str(data_sample[1]))

directory = './data'
negative = 'Negative'
positive = 'Positive'

negative_file_path = os.path.join(directory,negative)
positive_file_path = os.path.join(directory,positive)

negative_files=[os.path.join(negative_file_path,file) for file in  os.listdir(negative_file_path) if file.endswith(".jpg")]
negative_files.sort()

positive_files=[os.path.join(positive_file_path,file) for file in  os.listdir(positive_file_path) if file.endswith(".jpg")]
positive_files.sort()

number_of_sample_p = len(positive_files)
number_of_sample_n = len(negative_files)
number_of_samples = number_of_sample_p + number_of_sample_n
print(f"p:{number_of_sample_p} n:{number_of_sample_n} t:{number_of_samples}")

Y=torch.zeros([number_of_samples])
Y=Y.type(torch.LongTensor)
Y.type()

Y[::2]=1 # create label odd and even
Y[1::2]=0

def combine_dataset(p,n,total_num):
    result = []
    for i in range(total_num):
        if i %2 == 0:
            result.append(n[i//2])
        else:
            result.append(p[i//2])

    return result

print(number_of_samples)
all_files = combine_dataset(positive_files,negative_files,number_of_samples)

for y,file in zip(Y, all_files[0:4]):
    plt.imshow(Image.open(file))
    plt.title("y="+str(y.item()))
    plt.show()




    
