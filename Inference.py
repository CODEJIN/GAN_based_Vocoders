import torch
import numpy as np
import logging, yaml, os, sys, argparse, time, importlib, math
from tqdm import tqdm
from collections import defaultdict
from tensorboardX import SummaryWriter
import matplotlib
matplotlib.use('agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
from scipy.io import wavfile

from Modules.Commons import MultiResolutionSTFTLoss

with open('./Hyper_Parameters/Commons.yaml') as f:
    hp_Dict = yaml.load(f, Loader=yaml.Loader)

Generator = importlib.import_module('Modules.{}'.format(hp_Dict['Generator'])).Generator

with open('./Hyper_Parameters/{}.yaml'.format(hp_Dict['Generator'])) as f:
    hp_Dict['Generator'] = yaml.load(f, Loader=yaml.Loader)['Generator']

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")


class Dataset(torch.utils.data.Dataset):
    def __init__(self, mel_Path, stop_Path= None):
        super(Dataset, self).__init__()
        self.mel_Path = mel_Path.replace('\\', '/')
        self.stop_Path = stop_Path.replace('\\', '/') if not stop_Path is None else stop_Path

        self.pattern_List = [
            os.path.join(root, file).replace('\\', '/')
            for root, _, files in os.walk(mel_Path)
            for file in files
            if os.path.splitext(file)[1].upper() == '.NPY'
            ]

    def __getitem__(self, idx):
        mel = np.load(self.pattern_List[idx])
        mel_Path = self.pattern_List[idx]
        stop = None
        if not self.stop_Path is None:
            stop_Path = mel_Path.replace(self.mel_Path, self.stop_Path)
            stop = np.load(stop_Path) if os.path.exists(stop_Path) else None

        return mel, stop, mel_Path

    def __len__(self):
        return len(self.pattern_List)

class Collater:
    def __init__(self):
        try: self.upsample_Pad = hp_Dict['Generator']['Upsample']['Pad']
        except KeyError: self.upsample_Pad = 0

    def __call__(self, batch):
        mels, stops, files = zip(*batch)
        lengths = [mel.shape[0] * hp_Dict['Sound']['Frame_Shift'] for mel in mels]
        
        mels = [
            np.clip(mel, -hp_Dict['Sound']['Max_Abs_Mel'], hp_Dict['Sound']['Max_Abs_Mel'])
            for mel in mels
            ]
        mels = [
            np.pad(mel, [[self.upsample_Pad, self.upsample_Pad], [0, 0]], 'reflect')
            for mel in mels
            ]
        max_Length = max([mel.shape[0] for mel in mels])
        mels = [
            np.pad(mel, [[0, max_Length - mel.shape[0]], [0, 0]], constant_values= -hp_Dict['Sound']['Max_Abs_Mel'])
            for mel in mels
            ]
        mels = np.stack(mels, axis= 0)
        
        mels = torch.FloatTensor(mels).transpose(2, 1)   # [Batch, Mel_dim, Time]
        noises = torch.randn(mels.size(0), max(lengths)) # [Batch, Time]
        
        return mels, noises, lengths, stops, files


class Inferencer:
    def __init__(self, checkpoint_Path):
        self.Model_Generate()
        self.Load_Checkpoint(checkpoint_Path= checkpoint_Path)

    def Model_Generate(self):
        self.model = Generator(mel_dims= hp_Dict['Sound']['Mel_Dim']).to(device)
        self.model.eval()

    @torch.no_grad()
    def Inference_Step(self, mels, noises, lengths, stops, files, result_Path):
        mels = mels.to(device)
        noises = noises.to(device)
        
        fakes = self.model(noises, mels).cpu().numpy()
        for mel, fake, stop, file in zip(mels.cpu().numpy(), fakes, stops, files):
            if not stop is None and any(stop < 0.5):
                slice_Index = np.argmax(stop < 0.5)
            else:
                slice_Index = mel.shape[1] - 2 * hp_Dict['Generator']['Upsample']['Pad']
            mel = mel[:, hp_Dict['Generator']['Upsample']['Pad']:slice_Index + hp_Dict['Generator']['Upsample']['Pad']]
            fake = fake[:slice_Index * hp_Dict['Sound']['Frame_Shift']]

            new_Figure = plt.figure(figsize=(80, 10 * 2), dpi=100)
            plt.subplot(211)
            plt.imshow(mel, aspect='auto', origin='lower')
            plt.title('Mel spectrogram    File: {}'.format(file))
            plt.colorbar()
            plt.subplot(212)
            plt.plot(fake)
            plt.title('Inference wav    File: {}'.format(file))
            plt.margins(x= 0)
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(
                os.path.join(result_Path, 'PNG', os.path.basename(file)[:-4] + '.png').replace("\\", "/")
                )
            plt.close(new_Figure)

            wavfile.write(
                filename= os.path.join(result_Path, 'WAV', os.path.basename(file)[:-4] + '.wav').replace("\\", "/"),
                data= (fake * 32767.5).astype(np.int16),
                rate= hp_Dict['Sound']['Sample_Rate']
                )

    def Inference(
        self,
        mel_Path,
        stop_Path= None,
        result_Path= './results',
        batch_Size= hp_Dict['Inference_Batch_Size']
        ):
        logging.info('Mel-spectrogram path: {}'.format(mel_Path))
        logging.info('Result save path: {}'.format(result_Path))
        logging.info('Start inference.')

        dataLoader = torch.utils.data.DataLoader(
            dataset= Dataset(mel_Path, stop_Path),
            shuffle= False,
            collate_fn= Collater(),
            batch_size= batch_Size,
            num_workers= hp_Dict['Train']['Num_Workers'],
            pin_memory= True
            )

        os.makedirs(os.path.join(result_Path, 'PNG').replace('\\', '/'), exist_ok= True)
        os.makedirs(os.path.join(result_Path, 'WAV').replace('\\', '/'), exist_ok= True)
        for mels, noises, lengths, stops, files in tqdm(dataLoader, desc='[Inference]'):
            self.Inference_Step(mels, noises, lengths, stops, files, result_Path)

    def Load_Checkpoint(self, checkpoint_Path):
        state_Dict = torch.load(
            checkpoint_Path,
            map_location= 'cpu'
            )
        self.model.load_state_dict(state_Dict['Model']['Generator'])
        self.model.remove_weight_norm()

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-c', '--checkpoint', required= True)
    argParser.add_argument('-m', '--mel', required= True)
    argParser.add_argument('-s', '--stop', default= None)
    argParser.add_argument('-r', '--result', default='./results')
    argParser.add_argument('-b', '--batch', default= hp_Dict['Inference_Batch_Size'])
    argParser.add_argument('-gpu', '--gpu', default='-1')
    args = argParser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES']= args.gpu

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(0)
    
    new_Inferencer = Inferencer(checkpoint_Path= args.checkpoint)
    new_Inferencer.Inference(
        mel_Path= args.mel,
        stop_Path= args.stop,
        result_Path= args.result
        )

# python Inference.py -c "D:/GAN_based_Vocoder.Results/PWGAN.SR24K.Results.VCTKLibri/Checkpoint/S_400000.pkl" -m "D:\GoogleDrive\Colab_Test\Pitchtron\Pitchtron.Results\SR24K.Results.Debugging.LSSMA.GST.LJVCTK\Inference\Step-26000\NPY" -r D:/TTT -gpu 0
# python Inference.py -c "D:/GAN_based_Vocoder.Results/PWGAN.SR24K.Results.VCTKLibri/Checkpoint/S_400000.pkl" -m "D:\GoogleDrive\Colab_Test\Pitchtron\Pitchtron.Results\SR24K.Results.Debugging.LSSMA.Pitchtron.LJVCTK.ADV_01\Inference\Step-34000\NPY\Mel" -r D:/Pitchtron.LSSAM -gpu 0