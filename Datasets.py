import torch
import numpy as np
import yaml, pickle, os, math, logging
from random import shuffle

from Pattern_Generator import Pattern_Generate

with open('./Hyper_Parameters/Commons.yaml') as f:
    hp_Dict = yaml.load(f, Loader=yaml.Loader)

class Train_Dataset(torch.utils.data.Dataset):
    def __init__(self, is_Eval):
        super(Train_Dataset, self).__init__()
        self.dataset_Type = 'Eval' if is_Eval else 'Train'
            

        metadata_Dict = pickle.load(open(
            os.path.join(
                hp_Dict['Train']['{}_Pattern'.format(self.dataset_Type)]['Path'],
                hp_Dict['Train']['{}_Pattern'.format(self.dataset_Type)]['Metadata_File']
                ).replace('\\', '/'), 'rb'
            ))
        self.file_List = metadata_Dict['File_List']
        if not is_Eval:
            self.file_List *= hp_Dict['Train']['Train_Pattern']['Accumulated_Dataset_Epoch']
            
        self.cache_Dict = {}

    def __getitem__(self, idx):
        if idx in self.cache_Dict.keys():
            return self.cache_Dict[idx]

        file = self.file_List[idx]
        path = os.path.join(hp_Dict['Train']['{}_Pattern'.format(self.dataset_Type)]['Path'], file).replace('\\', '/')
        pattern_Dict = pickle.load(open(path, 'rb'))
        pattern = pattern_Dict['Audio'], pattern_Dict['Mel']

        if hp_Dict['Train']['Use_Pattern_Cache']:
            self.cache_Dict[path] = pattern
        
        return pattern

    def __len__(self):
        return len(self.file_List)

class Inference_Dataset(torch.utils.data.Dataset):
    def __init__(self, pattern_path= 'Inference_Wav_for_Training.txt', top_db= 30):
        super(Inference_Dataset, self).__init__()
        self.top_dB = 30

        self.pattern_List = []
        for line in open(pattern_path, 'r').readlines()[1:]:
            label, path = line.strip().split('\t')            
            self.pattern_List.append((label, path))

        self.cache_Dict = {}

    def __getitem__(self, idx):
        if idx in self.cache_Dict.keys():
            return self.cache_Dict[idx]

        label, path = self.pattern_List[idx]
        audio, mel, _ = Pattern_Generate(path, top_db= self.top_dB)
        pattern = audio, mel, label

        if hp_Dict['Train']['Use_Pattern_Cache']:
            self.cache_Dict[idx] = pattern
 
        return pattern

    def __len__(self):
        return len(self.pattern_List)


class Train_Collater:
    def __init__(self, upsample_Pad):
        self.upsample_Pad = upsample_Pad
        self.mel_Length = hp_Dict['Train']['Wav_Length'] // hp_Dict['Sound']['Frame_Shift']

    def __call__(self, batch):
        audios, mels = self.Stack(*zip(*batch))

        audios = torch.FloatTensor(audios)   # [Batch, Time]
        mels = torch.FloatTensor(mels).transpose(2, 1)   # [Batch, Time, Mel_dim] -> [Batch, Mel_dim, Time]
        noises = torch.randn(size= audios.size()) # [Batch, Time]

        return audios, mels, noises

    def Stack(self, audios, mels):
        audio_List = []
        mel_List = []
        for audio, mel in zip(audios, mels):
            mel_Pad = max(0, self.mel_Length + 2 * self.upsample_Pad - mel.shape[0])
            audio_Pad = max(0, hp_Dict['Train']['Wav_Length'] + 2 * self.upsample_Pad * hp_Dict['Sound']['Frame_Shift'] - audio.shape[0])
            mel = np.pad(
                mel,
                [[int(np.floor(mel_Pad / 2)), int(np.ceil(mel_Pad / 2))], [0, 0]],
                mode= 'reflect'
                )
            audio = np.pad(
                audio,
                [int(np.floor(audio_Pad / 2)), int(np.ceil(audio_Pad / 2))],
                mode= 'reflect'
                )

            mel_Offset = np.random.randint(self.upsample_Pad, max(mel.shape[0] - (self.mel_Length + self.upsample_Pad), self.upsample_Pad + 1))
            audio_Offset = mel_Offset * hp_Dict['Sound']['Frame_Shift']
            mel = mel[mel_Offset - self.upsample_Pad:mel_Offset + self.mel_Length + self.upsample_Pad]
            audio = audio[audio_Offset:audio_Offset + hp_Dict['Train']['Wav_Length']]

            audio_List.append(audio)
            mel_List.append(mel)

        return np.stack(audio_List, axis= 0), np.stack(mel_List, axis= 0)

class Inference_Collater:    
    def __init__(self, upsample_Pad):
        self.upsample_Pad = upsample_Pad
        self.mel_Length = hp_Dict['Train']['Wav_Length'] // hp_Dict['Sound']['Frame_Shift']

    def __call__(self, batch):
        max_Wav_Length = int(np.ceil(max([audio.shape[0] for audio, _, _ in batch]) / hp_Dict['Sound']['Frame_Shift']) * hp_Dict['Sound']['Frame_Shift'])
        max_Mel_Length = max_Wav_Length // hp_Dict['Sound']['Frame_Shift']

        audios, mels, lengths, labels = [], [], [], []
        for audio, mel, label in batch:
            length = audio.shape[0]

            audio = np.pad(
                audio,
                pad_width=[0, max_Wav_Length - audio.shape[0]],
                constant_values= 0
                )
            mel = np.pad(
                mel,
                pad_width=[[self.upsample_Pad, max_Mel_Length - mel.shape[0] + self.upsample_Pad], [0, 0]],
                constant_values= -hp_Dict['Sound']['Max_Abs_Mel']
                )
            
            audios.append(audio)
            mels.append(mel)
            lengths.append(length)
            labels.append(label)
            
        audios = torch.FloatTensor(np.stack(audios, axis= 0))   # [Batch, Time]
        mels = torch.FloatTensor(np.stack(mels, axis= 0)).transpose(2, 1)   # [Batch, Time, Mel_dim] -> [Batch, Mel_dim, Time]
        noises = torch.randn(size= audios.size()) # [Batch, Time]
        
        return audios, mels, noises, lengths, labels