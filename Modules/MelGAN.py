import torch
import numpy as np
from collections import OrderedDict
import yaml, math, logging

with open('./Hyper_Parameters/MelGAN.yaml') as f:
    hp_Dict = yaml.load(f, Loader=yaml.Loader)

class Generator(torch.nn.Module):
    def __init__(self, mel_dims):
        super(Generator, self).__init__()

        self.layer = torch.nn.Sequential()
        self.layer.add_module('First', torch.nn.Conv1d(
            in_channels= mel_dims,
            out_channels= hp_Dict['Generator']['Channels'],
            kernel_size= hp_Dict['Generator']['Kernel_Size'],
            padding= (hp_Dict['Generator']['Kernel_Size'] - 1) // 2,
            bias= True
            ))

        for index, scale in enumerate(hp_Dict['Generator']['Upsample_Scales']):
            self.layer.add_module('LeakyReLU_{}'.format(index), torch.nn.LeakyReLU(0.2))
            self.layer.add_module('Conv1d_Transpose_{}'.format(index), torch.nn.ConvTranspose1d(
                in_channels= hp_Dict['Generator']['Channels'] // 2 ** index,
                out_channels= hp_Dict['Generator']['Channels'] // 2 ** (index + 1),
                kernel_size= scale * 2,
                stride= scale,
                padding= scale // 2 + scale % 2,
                output_padding= scale % 2,
                bias= True                
                ))
            for stack_Index in range(hp_Dict['Generator']['Residual_Stack']['Stacks']):
                self.layer.add_module('Residual_Stack_{}_{}'.format(index, stack_Index), Residual_Stack(
                    channels= hp_Dict['Generator']['Channels'] // 2 ** (index + 1),
                    kernel_size = hp_Dict['Generator']['Residual_Stack']['Kernel_Size'],
                    dilation= hp_Dict['Generator']['Residual_Stack']['Kernel_Size'] ** stack_Index
                    ))

        self.layer.add_module('Last', torch.nn.Conv1d(
            in_channels= hp_Dict['Generator']['Channels'] // 2 ** (index + 1),
            out_channels= 1,
            kernel_size= hp_Dict['Generator']['Kernel_Size'],
            padding= (hp_Dict['Generator']['Kernel_Size'] - 1) // 2,
            bias= True
            ))
        self.layer.add_module('Tanh', torch.nn.Tanh())

        self.apply_weight_norm()
        self.reset_parameters()

    def forward(self, x):
        return self.layer(x).squeeze(1)


    def remove_weight_norm(self):
        def _remove_weight_norm(m):
            try:
                logging.debug(f'Weight norm is removed from {m}.')
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):                
                torch.nn.utils.weight_norm(m)
                logging.debug(f'Weight norm is applied to {m}.')

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):                
                m.weight.data.normal_(0.0, 0.2)
                logging.debug(f'Reset parameters in {m}.')

        self.apply(_reset_parameters)

class Residual_Stack(torch.nn.Module):
    def __init__(
        self,
        channels,
        kernel_size,
        dilation,
        ):
        super(Residual_Stack, self).__init__()

        self.layer_Dict = torch.nn.ModuleDict()
        self.layer_Dict['Stack'] = torch.nn.Sequential()
        
        self.layer_Dict['Stack'].add_module('LeakyReLU_0', torch.nn.LeakyReLU(0.2))
        self.layer_Dict['Stack'].add_module('Conv1d_0', torch.nn.Conv1d(
            in_channels= channels,
            out_channels= channels,
            kernel_size= kernel_size,
            padding= (kernel_size - 1) // 2 * dilation,
            dilation= dilation,
            bias= True
            ))
        self.layer_Dict['Stack'].add_module('LeakyReLU_1', torch.nn.LeakyReLU(0.2))
        self.layer_Dict['Stack'].add_module('Conv1d_1', torch.nn.Conv1d(
            in_channels= channels,
            out_channels= channels,
            kernel_size= kernel_size,   # kan-bayashi used 1... why?
            padding= (kernel_size - 1) // 2,
            bias= True
            ))

        self.layer_Dict['Skip'] = torch.nn.Conv1d(
            in_channels= channels,
            out_channels= channels,
            kernel_size= 1,
            bias= True
            )

    def forward(self, x):
        return self.layer_Dict['Stack'](x) + self.layer_Dict['Skip'](x)


class Single_Scale_Discriminator(torch.nn.Module):
    def __init__(self):
        super(Single_Scale_Discriminator, self).__init__()

        self.layer_List = torch.nn.ModuleList()
        previous_Channels = 1

        for index, (channels, kernel_size) in enumerate(zip(
            hp_Dict['Discriminator']['First']['Channels'],
            hp_Dict['Discriminator']['First']['Kernel_Size'],
            )):
            sequence = OrderedDict([
                ('Conv', torch.nn.Conv1d(
                    in_channels= previous_Channels,
                    out_channels= channels,
                    kernel_size= kernel_size,
                    padding= (kernel_size - 1) // 2,
                    bias= True
                    )),
                ('LeakyReLU', torch.nn.LeakyReLU(
                    negative_slope= 0.2,
                    inplace= True
                    ))
                ])
            self.layer_List.add_module('First_{}'.format(index), torch.nn.Sequential(sequence))            
            previous_Channels = channels

        for index, (channels, kernel_size, stride, group) in enumerate(zip(
            hp_Dict['Discriminator']['Downsample']['Channels'],
            hp_Dict['Discriminator']['Downsample']['Kernel_Size'],
            hp_Dict['Discriminator']['Downsample']['Stride'],
            hp_Dict['Discriminator']['Downsample']['Group'],
            )):
            sequence = OrderedDict([
                ('Conv', torch.nn.Conv1d(
                    in_channels= previous_Channels,
                    out_channels= channels,
                    kernel_size= kernel_size,
                    stride= stride,
                    padding= stride * 5,
                    groups= group,
                    bias= True
                    )),
                ('LeakyReLU', torch.nn.LeakyReLU(
                    negative_slope= 0.2,
                    inplace= True
                    ))
                ])
            self.layer_List.add_module('Downsample_{}'.format(index), torch.nn.Sequential(sequence))
            previous_Channels = channels

        for index, (channels, kernel_size) in enumerate(zip(
            hp_Dict['Discriminator']['Last']['Channels'],
            hp_Dict['Discriminator']['Last']['Kernel_Size'],
            )):
            sequence = OrderedDict([
                ('Conv', torch.nn.Conv1d(
                    in_channels= previous_Channels,
                    out_channels= channels,
                    kernel_size= kernel_size,
                    padding= (kernel_size - 1) // 2,
                    bias= True
                    ))
                ])
            if index < len(hp_Dict['Discriminator']['Last']['Channels']) - 1:                
                sequence.update({
                    'LeakyReLU': torch.nn.LeakyReLU(
                        negative_slope= 0.2,
                        inplace= True
                        )
                    })
                sequence.move_to_end('LeakyReLU', last= True)
            self.layer_List.add_module('Last_{}'.format(index), torch.nn.Sequential(sequence))
            previous_Channels = channels

    def forward(self, x):
        outputs = []
        x = x.unsqueeze(1)
        for layer in self.layer_List:
            x = layer(x)
            outputs.append(x)
        
        return outputs

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layer_Dict = torch.nn.ModuleDict()
        for index in range(hp_Dict['Discriminator']['Count']):
            self.layer_Dict['Discriminator_{}'.format(index)] = Single_Scale_Discriminator()

        self.layer_Dict['Pooling'] = torch.nn.AvgPool1d(
            kernel_size= hp_Dict['Discriminator']['Pooling']['Kernel_Size'],
            stride= hp_Dict['Discriminator']['Pooling']['Stride'],
            padding= hp_Dict['Discriminator']['Pooling']['Padding'],
            count_include_pad= False
            )

        self.apply_weight_norm()
        self.reset_parameters()

    def forward(self, x):
        outputs = []
        for index in range(hp_Dict['Discriminator']['Count']):
            outputs.extend(self.layer_Dict['Discriminator_{}'.format(index)](x))
            x = self.layer_Dict['Pooling'](x.unsqueeze(1)).squeeze(1)   # It is better to construct that 'Discriminator' can be used independently of 'Multi_Scale_Discriminator'.

        return outputs

    def remove_weight_norm(self):
        def _remove_weight_norm(m):
            try:
                logging.debug(f'Weight norm is removed from {m}.')
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):                
                torch.nn.utils.weight_norm(m)
                logging.debug(f'Weight norm is applied to {m}.')

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):                
                m.weight.data.normal_(0.0, 0.2)
                logging.debug(f'Reset parameters in {m}.')

        self.apply(_reset_parameters)
