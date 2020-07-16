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
from Datasets import Train_Dataset, Inference_Dataset, Train_Collater, Inference_Collater
from Radam import RAdam

with open('./Hyper_Parameters/Commons.yaml') as f:
    hp_Dict = yaml.load(f, Loader=yaml.Loader)

Generator = importlib.import_module('Modules.{}'.format(hp_Dict['Generator'])).Generator
Discriminator = importlib.import_module('Modules.{}'.format(hp_Dict['Discriminator'])).Discriminator

with open('./Hyper_Parameters/{}.yaml'.format(hp_Dict['Generator'])) as f:
    hp_Dict['Generator'] = yaml.load(f, Loader=yaml.Loader)
with open('./Hyper_Parameters/{}.yaml'.format(hp_Dict['Discriminator'])) as f:
    hp_Dict['Discriminator'] = yaml.load(f, Loader=yaml.Loader)


if not hp_Dict['Device'] is None:
    os.environ['CUDA_VISIBLE_DEVICES']= hp_Dict['Device']

if not torch.cuda.is_available():
    device = torch.device('cpu')
else:
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(0)

logging.basicConfig(
        level=logging.INFO, stream=sys.stdout,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

if torch.cuda.is_available() and hp_Dict['Use_Mixed_Precision']: 
    try:
        from apex import amp
    except:
        logging.info('There is no apex modules in the environment. Mixed precision does not work.')
        hp_Dict['Use_Mixed_Precision'] = False

class Trainer:
    def __init__(self, steps= 0):
        self.steps = steps
        self.epochs = 0

        self.Datset_Generate()
        self.Model_Generate()

        self.scalar_Dict = {
            'Train': defaultdict(float),
            'Evaluation': defaultdict(float),
            }
        self.writer_Dict = {
            'Train': SummaryWriter(os.path.join(hp_Dict['Log_Path'], 'Train')),
            'Evaluation': SummaryWriter(os.path.join(hp_Dict['Log_Path'], 'Evaluation')),
            }

        self.Load_Checkpoint()

    def Datset_Generate(self):
        train_Dataset = Train_Dataset(is_Eval= False)
        dev_Dataset = Train_Dataset(is_Eval= True)
        inference_Dataset = Inference_Dataset()
        logging.info('The number of train patterns = {}.'.format(len(train_Dataset)))
        logging.info('The number of development patterns = {}.'.format(len(dev_Dataset)))
        logging.info('The number of inference patterns = {}.'.format(len(inference_Dataset)))

        try:
            upsample_Pad = hp_Dict['Generator']['Upsample']['Pad']
        except:
            upsample_Pad = 0

        collater = Train_Collater(upsample_Pad= upsample_Pad)
        inference_Collater = Inference_Collater(upsample_Pad= upsample_Pad)

        self.dataLoader_Dict = {}
        self.dataLoader_Dict['Train'] = torch.utils.data.DataLoader(
            dataset= train_Dataset,
            shuffle= True,
            collate_fn= collater,
            batch_size= hp_Dict['Train']['Batch_Size'],
            num_workers= hp_Dict['Train']['Num_Workers'],
            pin_memory= True
            )
        self.dataLoader_Dict['Dev'] = torch.utils.data.DataLoader(
            dataset= dev_Dataset,
            shuffle= False,
            collate_fn= collater,
            batch_size= hp_Dict['Train']['Batch_Size'],
            num_workers= hp_Dict['Train']['Num_Workers'],
            pin_memory= True
            )
        self.dataLoader_Dict['Inference'] = torch.utils.data.DataLoader(
            dataset= inference_Dataset,
            shuffle= False,
            collate_fn= inference_Collater,
            batch_size= hp_Dict['Inference_Batch_Size'] or hp_Dict['Train']['Batch_Size'],
            num_workers= hp_Dict['Train']['Num_Workers'],
            pin_memory= True
            )

    def Model_Generate(self):
        self.model_Dict = {
            'Generator': Generator(mel_dims= hp_Dict['Sound']['Mel_Dim']).to(device),
            'Discriminator': Discriminator().to(device)
            }
        self.criterion_Dict = {
            'STFT': MultiResolutionSTFTLoss(
                fft_sizes= hp_Dict['STFT_Loss_Resolution']['FFT_Sizes'],
                shift_lengths= hp_Dict['STFT_Loss_Resolution']['Shfit_Lengths'],
                win_lengths= hp_Dict['STFT_Loss_Resolution']['Win_Lengths'],
                ).to(device),
            'MSE': torch.nn.MSELoss().to(device)
            }
        self.optimizer_Dict = {
            'Generator': RAdam(
                params= self.model_Dict['Generator'].parameters(),
                lr= hp_Dict['Train']['Learning_Rate']['Generator']['Initial'],
                eps= hp_Dict['Train']['Learning_Rate']['Generator']['Epsilon'],
                ),
            'Discriminator': RAdam(
                params= self.model_Dict['Discriminator'].parameters(),
                lr= hp_Dict['Train']['Learning_Rate']['Discriminator']['Initial'],
                eps= hp_Dict['Train']['Learning_Rate']['Discriminator']['Epsilon'],
                )
            }
        self.scheduler_Dict = {
            'Generator': torch.optim.lr_scheduler.StepLR(
                optimizer= self.optimizer_Dict['Generator'],
                step_size= hp_Dict['Train']['Learning_Rate']['Generator']['Decay_Step'],
                gamma= hp_Dict['Train']['Learning_Rate']['Generator']['Decay_Rate'],
                ),
            'Discriminator': torch.optim.lr_scheduler.StepLR(
                optimizer= self.optimizer_Dict['Discriminator'],
                step_size= hp_Dict['Train']['Learning_Rate']['Discriminator']['Decay_Step'],
                gamma= hp_Dict['Train']['Learning_Rate']['Discriminator']['Decay_Rate'],
                )
            }

        if torch.cuda.is_available() and hp_Dict['Use_Mixed_Precision']: 
            amp_Wrapped = amp.initialize(
                models=[self.model_Dict['Generator'], self.model_Dict['Discriminator']],
                optimizers=[self.optimizer_Dict['Generator'], self.optimizer_Dict['Discriminator']]
                )
            self.model_Dict['Generator'], self.model_Dict['Discriminator'] = amp_Wrapped[0]
            self.optimizer_Dict['Generator'], self.optimizer_Dict['Discriminator'] = amp_Wrapped[1]
        
        logging.info(self.model_Dict['Generator'])
        logging.info(self.model_Dict['Discriminator'])

    def Train_Step(self, audios, mels, noises):
        loss_Dict = {}

        audios = audios.to(device)
        mels = mels.to(device)
        noises = noises.to(device)
        
        fakes = self.model_Dict['Generator'](noises, mels)
        
        loss_Dict['Spectral_Convergence'], loss_Dict['Magnitude'] = self.criterion_Dict['STFT'](fakes, audios)
        loss_Dict['Generator'] = loss_Dict['Spectral_Convergence'] + loss_Dict['Magnitude']
        if self.steps > hp_Dict['Train']['Discriminator_Delay']:
            fake_Discriminations = self.model_Dict['Discriminator'](fakes)
            if not isinstance(fake_Discriminations, list):
                fake_Discriminations = [fake_Discriminations]
            loss_Dict['Adversarial'] = 0.0
            for discrimination in fake_Discriminations:
                loss_Dict['Adversarial'] += self.criterion_Dict['MSE'](
                    discrimination,
                    discrimination.new_ones(discrimination.size())
                    )
            loss_Dict['Generator'] += hp_Dict['Train']['Adversarial_Weight'] * loss_Dict['Adversarial']
        
        
        self.optimizer_Dict['Generator'].zero_grad()
        if torch.cuda.is_available() and hp_Dict['Use_Mixed_Precision']: 
            with amp.scale_loss(loss_Dict['Generator'], self.optimizer_Dict['Generator']) as scaled_loss:
                scaled_loss.backward()
        else:
            loss_Dict['Generator'].backward()
        torch.nn.utils.clip_grad_norm_(
            parameters= amp.master_params(self.model_Dict['Generator'].parameters()),
            max_norm= hp_Dict['Train']['Generator_Gradient_Norm']
            )
        self.optimizer_Dict['Generator'].step()
        self.scheduler_Dict['Generator'].step()
                
        if self.steps > hp_Dict['Train']['Discriminator_Delay']:
            real_Discriminations = self.model_Dict['Discriminator'](audios)
            fake_Discriminations = self.model_Dict['Discriminator'](fakes.detach())   
            if not isinstance(real_Discriminations, list):
                real_Discriminations = [real_Discriminations]
            if not isinstance(fake_Discriminations, list):
                fake_Discriminations = [fake_Discriminations]

            loss_Dict['Real'] = 0.0
            for discrimination in real_Discriminations:
                loss_Dict['Real'] += self.criterion_Dict['MSE'](
                    discrimination,
                    discrimination.new_ones(discrimination.size())
                    )
            loss_Dict['Fake'] = 0.0
            for discrimination in fake_Discriminations:
                loss_Dict['Fake'] += self.criterion_Dict['MSE'](
                    discrimination,
                    discrimination.new_zeros(discrimination.size())
                    )
            loss_Dict['Discriminator'] = loss_Dict['Real'] + loss_Dict['Fake']

            self.optimizer_Dict['Discriminator'].zero_grad()
            if torch.cuda.is_available() and hp_Dict['Use_Mixed_Precision']: 
                with amp.scale_loss(loss_Dict['Discriminator'], self.optimizer_Dict['Discriminator']) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_Dict['Discriminator'].backward()
            torch.nn.utils.clip_grad_norm_(
                parameters= amp.master_params(self.model_Dict['Discriminator'].parameters()),
                max_norm= hp_Dict['Train']['Discriminator_Gradient_Norm']
                )
            self.optimizer_Dict['Discriminator'].step()
            self.scheduler_Dict['Discriminator'].step()
        
        self.steps += 1
        self.tqdm.update(1)

        for tag, loss in loss_Dict.items():
            self.scalar_Dict['Train']['Loss/{}'.format(tag)] += loss

    def Train_Epoch(self):
        for audios, mels, noises in self.dataLoader_Dict['Train']:
            self.Train_Step(audios, mels, noises)
            
            if self.steps % hp_Dict['Train']['Checkpoint_Save_Interval'] == 0:
                self.Save_Checkpoint()

            if self.steps % hp_Dict['Train']['Logging_Interval'] == 0:
                self.scalar_Dict['Train'] = {
                    tag: loss / hp_Dict['Train']['Logging_Interval']
                    for tag, loss in self.scalar_Dict['Train'].items()
                    }
                self.scalar_Dict['Train']['Learning_Rate/Generator'] = self.scheduler_Dict['Generator'].get_last_lr()
                if self.steps >= hp_Dict['Train']['Discriminator_Delay']:
                    self.scalar_Dict['Train']['Learning_Rate/Discriminator'] = self.scheduler_Dict['Discriminator'].get_last_lr()
                self.Write_to_Tensorboard('Train', self.scalar_Dict['Train'])
                self.scalar_Dict['Train'] = defaultdict(float)

            if self.steps % hp_Dict['Train']['Evaluation_Interval'] == 0:
                self.Evaluation_Epoch()

            if self.steps % hp_Dict['Train']['Inference_Interval'] == 0:
                self.Inference_Epoch()
            
            if self.steps >= hp_Dict['Train']['Max_Step']:
                return

        self.epochs += 1


    @torch.no_grad()
    def Evaluation_Step(self, audios, mels, noises):
        loss_Dict = {}

        audios = audios.to(device)
        mels = mels.to(device)
        noises = noises.to(device)
        
        fakes = self.model_Dict['Generator'](noises, mels)
        
        loss_Dict['Spectral_Convergence'], loss_Dict['Magnitude'] = self.criterion_Dict['STFT'](fakes, audios)
        loss_Dict['Generator'] = loss_Dict['Spectral_Convergence'] + loss_Dict['Magnitude']
        if self.steps > hp_Dict['Train']['Discriminator_Delay']:
            fake_Discriminations = self.model_Dict['Discriminator'](fakes)
            if not isinstance(fake_Discriminations, list):
                fake_Discriminations = [fake_Discriminations]
            loss_Dict['Adversarial'] = 0.0
            for discrimination in fake_Discriminations:
                loss_Dict['Adversarial'] += self.criterion_Dict['MSE'](
                    discrimination,
                    discrimination.new_ones(discrimination.size())
                    )            
            loss_Dict['Generator'] += hp_Dict['Train']['Adversarial_Weight'] * loss_Dict['Adversarial']


        if self.steps > hp_Dict['Train']['Discriminator_Delay']:
            real_Discriminations = self.model_Dict['Discriminator'](audios)
            fake_Discriminations = self.model_Dict['Discriminator'](fakes.detach())   
            if not isinstance(real_Discriminations, list):
                real_Discriminations = [real_Discriminations]
            if not isinstance(fake_Discriminations, list):
                fake_Discriminations = [fake_Discriminations]

            loss_Dict['Real'] = 0.0
            for discrimination in real_Discriminations:
                loss_Dict['Real'] += self.criterion_Dict['MSE'](
                    discrimination,
                    discrimination.new_ones(discrimination.size())
                    )
            loss_Dict['Fake'] = 0.0
            for discrimination in fake_Discriminations:
                loss_Dict['Fake'] += self.criterion_Dict['MSE'](
                    discrimination,
                    discrimination.new_zeros(discrimination.size())
                    )
            loss_Dict['Discriminator'] = loss_Dict['Real'] + loss_Dict['Fake']

        for tag, loss in loss_Dict.items():
            self.scalar_Dict['Evaluation']['Loss/{}'.format(tag)] += loss

    def Evaluation_Epoch(self):
        logging.info('(Steps: {}) Start evaluation.'.format(self.steps))

        for model in self.model_Dict.values():
            model.eval()

        for step, (audios, mels, noises) in tqdm(
            enumerate(self.dataLoader_Dict['Dev'], 1),
            desc='[Evaluation]',
            total= math.ceil(len(self.dataLoader_Dict['Dev'].dataset) / hp_Dict['Train']['Batch_Size'])
            ):
            self.Evaluation_Step(audios, mels, noises)

        self.scalar_Dict['Evaluation'] = {
            tag: loss / step
            for tag, loss in self.scalar_Dict['Evaluation'].items()
            }
        self.Write_to_Tensorboard('Evaluation', self.scalar_Dict['Evaluation'])
        self.scalar_Dict['Evaluation'] = defaultdict(float)
        
        for model in self.model_Dict.values():
            model.train()


    @torch.no_grad()
    def Inference_Step(self, audios, mels, noises, lengths, labels, start_Index= 0, tag_Step= False, tag_Index= False):
        mels = mels.to(device)
        noises = noises.to(device)
        fakes = self.model_Dict['Generator'](noises, mels).cpu().numpy()

        os.makedirs(os.path.join(hp_Dict['Inference_Path'], 'Step-{}'.format(self.steps)).replace("\\", "/"), exist_ok= True)

        for index, (real, fake, length, label) in enumerate(zip(audios, fakes, lengths, labels)):
            real, fake = real[:length], fake[:length]
            new_Figure = plt.figure(figsize=(80, 10 * 2), dpi=100)
            plt.subplot(211)
            plt.plot(real)
            plt.title('Original wav    Label: {}    Index: {}'.format(label, index))
            plt.margins(x= 0)
            plt.subplot(212)
            plt.plot(fake)
            plt.title('Fake wav    Label: {}    Index: {}'.format(label, index))
            plt.margins(x= 0)
            plt.tight_layout()
            file = '{}{}{}'.format(
                'Step-{}.'.format(self.steps) if tag_Step else '',
                label,
                '.IDX_{}'.format(index + start_Index) if tag_Index else ''
                )
            plt.savefig(
                os.path.join(hp_Dict['Inference_Path'], 'Step-{}'.format(self.steps), '{}.PNG'.format(file)).replace("\\", "/")
                )
            plt.close(new_Figure)

            wavfile.write(
                filename= os.path.join(hp_Dict['Inference_Path'], 'Step-{}'.format(self.steps), '{}.WAV'.format(file)).replace("\\", "/"),
                data= (fake * 32767.5).astype(np.int16),
                rate= hp_Dict['Sound']['Sample_Rate']
                )

    def Inference_Epoch(self):
        logging.info('(Steps: {}) Start inference.'.format(self.steps))
        batches = hp_Dict['Inference_Batch_Size'] or hp_Dict['Train']['Batch_Size']

        for model in self.model_Dict.values():
            model.eval()

        for step, (audios, mels, noises, lengths, labels) in tqdm(
            enumerate(self.dataLoader_Dict['Inference'], 1),
            desc='[Inference]',
            total= math.ceil(len(self.dataLoader_Dict['Inference'].dataset) / batches)
            ):
            self.Inference_Step(audios, mels, noises, lengths, labels, start_Index= step * batches)

        for model in self.model_Dict.values():
            model.train()


    def Load_Checkpoint(self):
        if self.steps == 0:
            path = None
            for root, _, files in os.walk(hp_Dict['Checkpoint_Path']):
                path = max(
                    [os.path.join(root, file).replace('\\', '/') for file in files],
                    key = os.path.getctime
                    )
                break
            if path is None:
                return  # Initial training
        else:
            path = os.path.join(hp_Dict['Checkpoint_Path'], 'S_{}.pt'.format(self.steps).replace('\\', '/'))


        state_Dict = torch.load(path, map_location= 'cpu')

        self.model_Dict['Generator'].load_state_dict(state_Dict['Model']['Generator'])
        self.model_Dict['Discriminator'].load_state_dict(state_Dict['Model']['Discriminator'])
        
        self.optimizer_Dict['Generator'].load_state_dict(state_Dict['Optimizer']['Generator'])
        self.optimizer_Dict['Discriminator'].load_state_dict(state_Dict['Optimizer']['Discriminator'])

        self.scheduler_Dict['Generator'].load_state_dict(state_Dict['Scheduler']['Generator'])
        self.scheduler_Dict['Discriminator'].load_state_dict(state_Dict['Scheduler']['Discriminator'])
        
        self.steps = state_Dict['Steps']
        self.epochs = state_Dict['Epochs']

        if torch.cuda.is_available() and hp_Dict['Use_Mixed_Precision']: 
            if not 'AMP' in state_Dict.keys():
                logging.info('No AMP state dict is in the checkpoint. Model regards this checkpoint is trained without mixed precision.')
            else:                
                amp.load_state_dict(state_Dict['AMP'])

        logging.info('Checkpoint loaded at {} steps.'.format(self.steps))

    def Save_Checkpoint(self):
        os.makedirs(hp_Dict['Checkpoint_Path'], exist_ok= True)

        state_Dict = {
            'Model': {
                'Generator': self.model_Dict['Generator'].state_dict(),
                'Discriminator': self.model_Dict['Discriminator'].state_dict(),
                },
            'Optimizer': {
                'Generator': self.optimizer_Dict['Generator'].state_dict(),
                'Discriminator': self.optimizer_Dict['Discriminator'].state_dict(),
                },
            'Scheduler': {
                'Generator': self.scheduler_Dict['Generator'].state_dict(),
                'Discriminator': self.scheduler_Dict['Discriminator'].state_dict(),
                },
            'Steps': self.steps,
            'Epochs': self.epochs,
            }
        if torch.cuda.is_available() and hp_Dict['Use_Mixed_Precision']: 
            state_Dict['AMP'] = amp.state_dict()

        torch.save(
            state_Dict,
            os.path.join(hp_Dict['Checkpoint_Path'], 'S_{}.pkl'.format(self.steps).replace('\\', '/'))
            )

        logging.info('Checkpoint saved at {} steps.'.format(self.steps))


    def Train(self):
        self.tqdm = tqdm(
            initial= self.steps,
            total= hp_Dict['Train']['Max_Step'],
            desc='[Training]'
            )

        if hp_Dict['Train']['Initial_Inference']:
            self.Evaluation_Epoch()

        while self.steps < hp_Dict['Train']['Max_Step']:
            try:
                self.Train_Epoch()
            except KeyboardInterrupt:
                self.Save_Checkpoint()
                exit(1)
            
        self.tqdm.close()
        logging.info('Finished training.')

    def Write_to_Tensorboard(self, category, scalar_Dict):
        for tag, scalar in scalar_Dict.items():
            self.writer_Dict[category].add_scalar(tag, scalar, self.steps)

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-s', '--steps', default= 0, type= int)
    args = argParser.parse_args()
    
    new_Trainer = Trainer(steps= args.steps)    
    new_Trainer.Train()