import os
import sys
sys.path.append('.')
#sys.path.append('..')
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import numpy as np
import pandas as pd
import librosa
import time
import matplotlib.pyplot as plt
import copy
import os
from tqdm import tqdm
import torchaudio
import pandas as pd
from pydub import AudioSegment
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torchvision.models as models
import os
import time
import copy
from tqdm import tqdm
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='DataFramesPreparation')

parser.add_argument('--dataset_tsv_path', type=str, help='path to train.tsv and test.tsv', default ='./cv-corpus-7.0-2021-07-21/ru/' )
parser.add_argument('--train_name', type=str, help='filename.csv, that will be used for training', default ='train_part.csv' )
parser.add_argument('--val_name', type=str, help='filename.csv, that will be used for validation', default ='val_part.csv' )
parser.add_argument('--test_name', type=str, help='filename.csv, that will be used for testing', default ='test_dropna.csv' )
args = parser.parse_args()


def main():
    df_train = pd.read_csv(args.dataset_tsv_path+'train.tsv', sep='\t')
    df_test = pd.read_csv(args.dataset_tsv_path+'test.tsv', sep='\t')
    
    df_train = df_train[['client_id','path','gender']]
    df_test = df_test[['client_id','path','gender']]
    
    df_train_female = df_train[df_train.gender=='female'].copy()
    df_train_male = df_train[df_train.gender=='male'].sample(3000).copy()
    df_train = df_train_female.append(df_train_male)
    df_train = df_train.sample(frac=1)
    
    df_train = df_train.reset_index(drop=True)
    df_train['gender'].replace(to_replace=['female', 'male'],value= [0, 1], inplace=True)
    
    df_test = df_test.dropna()
    df_test['gender'].replace(to_replace=['female', 'male'],value= [0, 1], inplace=True)
    
    df_train.to_csv('train_preproc_balanced.csv')
    df_test.to_csv(args.test_name)
    
    df = pd.read_csv('train_preproc_balanced.csv')
    df.dtypes
    df = df.sample(5556)
    train_df = df[0:4556]
    val_df = df[4556:]
    
    train_df.to_csv(args.train_name)
    val_df.to_csv(args.val_name)
    
    
    
    

if __name__ == "__main__":
    main()
