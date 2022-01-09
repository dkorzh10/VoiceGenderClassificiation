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
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torchvision.models as models
import os
import time
from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics
import copy
from tqdm import tqdm
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='PyTorch Training')


parser.add_argument('--AUDIO_DIR', default="./cv-corpus-7.0-2021-07-21/ru/clips/", type=str,
                    help='path to directory with audio clips')

parser.add_argument('--ANN_FILE', default='train_preproc_balanced.csv', type=str,
                    help='path to directory with audio clips')

parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--train_name', type=str, help='filename.csv, that will be used for training', default ='train_part.csv' )
parser.add_argument('--val_name', type=str, help='filename.csv, that will be used for validation', default ='val_part.csv' )
parser.add_argument('--test_name', type=str, help='filename.csv, that will be used for testing', default ='test_dropna.csv' )
parser.add_argument('--batch_size', default=16, type=int,help='batchsize (default: 16)')
parser.add_argument('--num_workers', default=4, type=int,help='num dataloader workers (0 or greater)')

parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate', dest='lr')

parser.add_argument('--momentum', default=0.9, type=float,
                    help='sgd momentum',)

parser.add_argument('--epochs', default=3, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--best_model_name', type=str, help='filename (without .pt),to save best model on validation', default ='MelsSpecsResnetBest')
args = parser.parse_args()

#default values, will declared explicitly
AUDIO_DIR = args.AUDIO_DIR
ANN_FILE = args.ANN_FILE

if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
print(device) 

def train_model(model, dataloaders, criterion, optimizer, num_epochs=3, scheduler= None,
                savename='best_model', is_inception=False):
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase=='train':
                scheduler.step()
                
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), savename+'.pt')
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, val_acc_history




class CustomAudioDataset(Dataset):

    def __init__(self, audio_dir = AUDIO_DIR, annotation_file = ANN_FILE, resample_freq =16000,
                 transform=None, target_transform=None, n_mels = 64):
        df = pd.read_csv(annotation_file)
        self.audio_labels = df[['path', 'gender']].copy() #pd.read_csv(annotation_file, names=['paths', 'gender'])
        #print(self.audio_labels)
        self.audio_dir = audio_dir
        self.transform = transform
        self.target_transform = target_transform
        self.resample = resample_freq
        self.n_mels = n_mels
        
    def __len__(self):
        return len(self.audio_labels)
    
    def __getitem__(self, idx):
        
        label = self.audio_labels.iloc[idx, 1]
        #print(label)
        #label = int(label)
        label = torch.tensor(label)
        
        audio_path = os.path.join(self.audio_dir, self.audio_labels.iloc[idx, 0])
        waveform , sr = torchaudio.load(audio_path)

        
        if self.resample > 0:
            resample_transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.resample)
            waveform = resample_transform(waveform)
    
        melspectrogram_transform = torchaudio.transforms.MelSpectrogram(sample_rate=self.resample, n_mels=self.n_mels)
        melspectrogram = melspectrogram_transform(waveform)
        melspectogram_db = torchaudio.transforms.AmplitudeToDB()(melspectrogram)
        #print(melspectogram_db.shape)
        #Make sure all spectrograms are the same size
        fixed_length = 3 * (self.resample//100) #//200
        if melspectogram_db.shape[2] < fixed_length:
            melspectogram_db = torch.nn.functional.pad(
              melspectogram_db, (0, fixed_length - melspectogram_db.shape[2]))
        else:
            melspectogram_db = melspectogram_db[:, :, :fixed_length]

        return melspectogram_db, label #soundData, self.resample, melspectogram_db, self.labels[index]
        
#         return specgram, label

def test(model, criterion):
    test_dataset = CustomAudioDataset(annotation_file = args.test_name)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    model = model.eval()
    phase = 'test'
    running_loss = 0.0
    running_corrects = 0
    y = []
    y_pred_0 = []
    y_pred_1 = []
    for inputs, labels in tqdm(test_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        y.append(list(labels.view(-1).detach().cpu()))


        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            m = nn.Softmax(dim=1)
            q = m(outputs)

            y_pred_0.append(list(q[:,0].detach().cpu()))
            y_pred_1.append(list(q[:,1].detach().cpu()))
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)


        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(test_dataloader.dataset)
    epoch_acc = running_corrects.double() / len(test_dataloader.dataset)

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
    
    def flatten(t):
        return [float(item) for sublist in t for item in sublist]

    y = flatten(y)
    y_pred_1 = flatten(y_pred_1)
    
    print("ROC-AUC score on test dataset:", round(roc_auc_score(y, y_pred_1), 4))
    
    preds = y_pred_1
    fpr, tpr, threshold = metrics.roc_curve(y, preds)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('roc_auc.pdf')
    plt.show()

def train():
    
    
    train_dataset = CustomAudioDataset(annotation_file = args.train_name)
    val_dataset = CustomAudioDataset(annotation_file = args.val_name)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    
    
    classes =2 

    model = models.resnet18(pretrained=True)
    model.conv1=nn.Conv2d(1, model.conv1.out_channels, 
                          kernel_size=model.conv1.kernel_size[0], 
                          stride=model.conv1.stride[0], 
                          padding=model.conv1.padding[0])
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, classes)

    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum)
    #don't need scheduler, one can add arguments to parser
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 8, gamma = 0.1)
    criterion = nn.CrossEntropyLoss()
    
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    
    model, val_acc_history = train_model(model, dataloaders, criterion, optimizer, num_epochs=args.epochs,scheduler= scheduler, 
            savename=args.best_model_name, is_inception=False)
    

    test(model,criterion)
    
    
if __name__== "__main__":
    train()
    
    
