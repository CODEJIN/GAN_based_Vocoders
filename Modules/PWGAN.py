import torch
import torch.nn.functional as F
import numpy as np
import yaml, math, logging

with open('./Hyper_Parameters/PWGAN.yaml') as f:
    hp_Dict = yaml.load(f, Loader=yaml.Loader)

class Generator(torch.nn.Module):
    def __init__(self, mel_dims):
        super(Generator, self).__init__()

        self.layer_Dict = torch.nn.ModuleDict()
        self.layer_Dict['First'] = torch.nn.Sequential()
        self.layer_Dict['First'].add_module('Unsqueeze', Unsqueeze(dim= 1))
        self.layer_Dict['First'].add_module('Conv', Conv1d1x1(
            in_channels= 1,
            out_channels= hp_Dict['Generator']['Residual_Channels'],
            bias= True
            ))
        
        for block_Index in range(hp_Dict['Generator']['ResConvGLU']['Blocks']):
            for stack_Index in range(hp_Dict['Generator']['ResConvGLU']['Stacks_in_Block']):
                self.layer_Dict['ResConvGLU_{}_{}'.format(block_Index, stack_Index)] = ResConvGLU(
                    residual_channels= hp_Dict['Generator']['Residual_Channels'],
                    gate_channels= hp_Dict['Generator']['ResConvGLU']['Gate_Channels'],
                    skip_channels= hp_Dict['Generator']['ResConvGLU']['Skip_Channels'],
                    aux_channels= mel_dims,
                    kernel_size= hp_Dict['Generator']['ResConvGLU']['Kernel_Size'],
                    dilation= 2 ** stack_Index,
                    dropout= hp_Dict['Generator']['ResConvGLU']['Dropout_Rate'],
                    bias= True
                    )

        self.layer_Dict['Last'] = torch.nn.Sequential()
        self.layer_Dict['Last'].add_module('ReLU_0', torch.nn.ReLU(inplace= True))
        self.layer_Dict['Last'].add_module('Conv_0', Conv1d1x1(
            in_channels= hp_Dict['Generator']['ResConvGLU']['Skip_Channels'],
            out_channels= hp_Dict['Generator']['ResConvGLU']['Skip_Channels'],
            bias= True
            ))
        self.layer_Dict['Last'].add_module('ReLU_1', torch.nn.ReLU(inplace= True))
        self.layer_Dict['Last'].add_module('Conv_1', Conv1d1x1(
            in_channels= hp_Dict['Generator']['ResConvGLU']['Skip_Channels'],
            out_channels= 1,
            bias= True
            ))  #[Batch, 1, Time]
        self.layer_Dict['Last'].add_module('Squeeze', Squeeze(dim= 1)) #[Batch, Time]

        self.layer_Dict['Upsample'] = UpsampleNet(mel_dims)

        self.apply_weight_norm()
        
    def forward(self, x, auxs):        
        auxs = self.layer_Dict['Upsample'](auxs)

        x = self.layer_Dict['First'](x)
        skips = 0
        for block_Index in range(hp_Dict['Generator']['ResConvGLU']['Blocks']):
            for stack_Index in range(hp_Dict['Generator']['ResConvGLU']['Stacks_in_Block']):
                x, new_Skips = self.layer_Dict['ResConvGLU_{}_{}'.format(block_Index, stack_Index)](x, auxs)
                skips += new_Skips
        skips *= math.sqrt(1.0 / (hp_Dict['Generator']['ResConvGLU']['Blocks'] * hp_Dict['Generator']['ResConvGLU']['Stacks_in_Block']))

        logits = self.layer_Dict['Last'](skips)

        return logits

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
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):                
                torch.nn.utils.weight_norm(m)
                logging.debug(f'Weight norm is applied to {m}.')

        self.apply(_apply_weight_norm)

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layer = torch.nn.Sequential()
        self.layer.add_module('Unsqueeze', Unsqueeze(dim= 1))

        previous_Channels = 1        
        for index in range(hp_Dict['Discriminator']['Stacks'] - 1):
            dilation = max(1, index)
            padding = (hp_Dict['Discriminator']['Kernel_Size'] - 1) // 2 * dilation
            self.layer.add_module('Conv_{}'.format(index), Conv1d(
                in_channels= previous_Channels,
                out_channels= hp_Dict['Discriminator']['Channels'],
                kernel_size= hp_Dict['Discriminator']['Kernel_Size'],
                padding= padding,
                dilation= dilation,
                bias= True
                ))
            self.layer.add_module('LeakyReLU_{}'.format(index),  torch.nn.LeakyReLU(
                negative_slope= 0.2,
                inplace= True
                ))
            previous_Channels = hp_Dict['Discriminator']['Channels']

        self.layer.add_module('Last', Conv1d(
            in_channels= previous_Channels,
            out_channels= 1,
            kernel_size= hp_Dict['Discriminator']['Kernel_Size'],
            padding= (hp_Dict['Discriminator']['Kernel_Size'] - 1) // 2,
            bias= True
            ))
        self.layer.add_module('Squeeze', Squeeze(dim= 1))

        self.apply_weight_norm()

    def forward(self, x):
        return self.layer(x)

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f'Weight norm is applied to {m}.')

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        def _remove_weight_norm(m):
            try:
                logging.debug(f'Weight norm is removed from {m}.')
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)


class UpsampleNet(torch.nn.Module):
    def __init__(self, mel_dims):
        super(UpsampleNet, self).__init__()

        self.layer = torch.nn.Sequential()        
        self.layer.add_module('First', Conv1d(
            in_channels= mel_dims,
            out_channels= mel_dims,
            kernel_size= hp_Dict['Generator']['Upsample']['Pad'] * 2 + 1,
            bias= False
            ))  # [Batch, Mel_dim, Time]
        self.layer.add_module('Unsqueeze', Unsqueeze(dim= 1))    # [Batch, 1, Mel_dim, Time]
        for index, scale in enumerate(hp_Dict['Generator']['Upsample']['Scales']):
            self.layer.add_module('Stretch_{}'.format(index), Stretch2d(scale, 1, mode='nearest'))  # [Batch, 1, Mel_dim, Scaled_Time]
            self.layer.add_module('Conv2d_{}'.format(index), Conv2d(
                in_channels= 1,
                out_channels= 1,
                kernel_size= (1, scale * 2 + 1),
                padding= (0, scale),
                bias= False
                ))  # [Batch, 1, Mel_dim, Scaled_Time]
        self.layer.add_module('Squeeze', Squeeze(dim= 1))    # [Batch, Mel_dim, Scaled_Time]

    def forward(self, x):        
        return self.layer(x)

class ResConvGLU(torch.nn.Module):
    def __init__(
        self,
        residual_channels,
        gate_channels,
        skip_channels,
        aux_channels,
        kernel_size,
        dilation= 1,
        dropout= 0.0,
        bias= True
        ):
        super(ResConvGLU, self).__init__()

        self.layer_Dict = torch.nn.ModuleDict()
        self.layer_Dict['Conv1d'] = torch.nn.Sequential()
        self.layer_Dict['Conv1d'].add_module('Dropout', torch.nn.Dropout(p= dropout))
        self.layer_Dict['Conv1d'].add_module('Conv1d', Conv1d(
            in_channels= residual_channels,
            out_channels= gate_channels,
            kernel_size= kernel_size,
            padding= (kernel_size - 1) // 2 * dilation,
            dilation= dilation,
            bias= bias
            ))

        self.layer_Dict['Aux'] = Conv1d1x1(
            in_channels= aux_channels,
            out_channels= gate_channels,
            bias= False
            )

        self.layer_Dict['Out'] = Conv1d1x1(
            in_channels= gate_channels // 2,
            out_channels= residual_channels,
            bias= bias
            )

        self.layer_Dict['Skip'] = Conv1d1x1(
            in_channels= gate_channels // 2,
            out_channels= skip_channels,
            bias= bias
            )

    def forward(self, audios, auxs):
        residuals = audios

        audios = self.layer_Dict['Conv1d'](audios)
        audios_Tanh, audios_Sigmoid = audios.split(audios.size(1) // 2, dim= 1)

        auxs = self.layer_Dict['Aux'](auxs)
        auxs_Tanh, auxs_Sigmoid = auxs.split(auxs.size(1) // 2, dim= 1)

        audios_Tanh = torch.tanh(audios_Tanh + auxs_Tanh)
        audios_Sigmoid = torch.sigmoid(audios_Sigmoid + auxs_Sigmoid)
        audios = audios_Tanh * audios_Sigmoid 

        outs = (self.layer_Dict['Out'](audios) + residuals) * math.sqrt(0.5)
        skips = self.layer_Dict['Skip'](audios)

        return outs, skips

 
class Stretch2d(torch.nn.Module):
    def __init__(self, x_scale, y_scale, mode= 'nearest'):
        super(Stretch2d, self).__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.mode= mode

    def forward(self, x):
        return F.interpolate(
            x,
            scale_factor=(self.y_scale, self.x_scale),
            mode= self.mode
            )

class Conv1d(torch.nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        if not self.bias is None:
            torch.nn.init.zeros_(self.bias)

class Conv1d1x1(Conv1d):
    def __init__(self, in_channels, out_channels, bias):
        super(Conv1d1x1, self).__init__(
            in_channels= in_channels,
            out_channels= out_channels,
            kernel_size= 1,
            padding= 0,
            dilation= 1,
            bias= bias
            )

class Conv2d(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        self.weight.data.fill_(1.0 / np.prod(self.kernel_size))
        if not self.bias is None:
            torch.nn.init.zeros_(self.bias)

class Squeeze(torch.nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.squeeze(x, dim= self.dim)

class Unsqueeze(torch.nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.unsqueeze(x, dim= self.dim)
