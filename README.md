# GAN_based_Vocoders

This code is a replication code of [kan-bayashi's code](https://github.com/kan-bayashi/ParallelWaveGAN). I wrote this code for my independent study about PyTorch. If you want to use Parallel WaveGAN model, I recommend that you refer to the original kan-bayashi's code.

The following is the paper I referred:
```
Oord, A. V. D., Dieleman, S., Zen, H., Simonyan, K., Vinyals, O., Graves, A., ... & Kavukcuoglu, K. (2016). Wavenet: A generative model for raw audio. arXiv preprint arXiv:1609.03499.
Yamamoto, R., Song, E., & Kim, J. M. (2020, May). Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 6199-6203). IEEE.
```

# Requirements
* torch >= 1.5.1
* tensorboardX >= 2.0
* librosa >= 0.7.2

* Optional    
    * tensorboard >= 2.2.2 (for loss flow)
    * apex >= 0.1 (for mixed precision)

# Used dataset
Currently uploaded code is compatible with the following datasets. The O mark to the left of the dataset name is the dataset actually used in the uploaded result.

```
[O] LJSpeech: https://keithito.com/LJ-Speech-Dataset/
[O] VCTK: https://datashare.is.ed.ac.uk/handle/10283/2651
[O] LibriSpeech: https://www.openslr.org/12
[X] TIMIT: http://academictorrents.com/details/34e2b78745138186976cbc27939b1b34d18bd5b3
[X] Blizzard Challenge 2013: http://www.cstr.ed.ac.uk/projects/blizzard/
[X] CMUArctic: http://www.festvox.org/cmu_arctic/index.html
```

* Currently uploaded code is compatible with the following datasets.
* The O marks to the left of the dataset name are the dataset actually used in the uploaded result.

| Using  | Dataset   | Dataset address                                 |
|--------|-----------|-------------------------------------------------|
| O      | LJSpeech  | https://keithito.com/LJ-Speech-Dataset/         |
| X      | BC2013    | http://www.cstr.ed.ac.uk/projects/blizzard/     |
| X      | CMU Arctic| http://www.festvox.org/cmu_arctic/index.html    |
| O      | VCTK      | https://datashare.is.ed.ac.uk/handle/10283/2651 |
| O      | LibriTTS  | https://openslr.org/60/                         |


# Hyper parameters
Before proceeding, please set the pattern, inference, and checkpoint paths in `Hyper_Parameters` according to your environment.

## Commons.yaml

This file set the base environment.

* Sound
    * Setting basic sound parameters.

* Generator
    * Setting the type of generator used
    * Current option is 'PWGAN' or 'MelGAN'

* Discriminator
    * Setting the type of generator used
    * Current option is 'PWGAN' or 'MelGAN'

* STFT_Loss_Resolution
    * Generator optimizer uses multi-scale STFT loss
    * Setting each STFT loss resolution.

* Train
    * Setting the parameters of training.    
    * Wav length must be a multiple of frame shift size of sound.

* Inference_Path
    * Setting the inference path

* Checkpoint_Path
    * Setting the checkpoint path

* Log_Path
    * Setting the tensorboard log path

* Use_Mixed_Precision
    * If true, mixed preicision is used.
    * This option requires `nividia apex` module.

* Device
    * Setting which GPU device is used in multi-GPU enviornment.
    * Or, if using only CPU, please set '-1'.

## PWGAN.yaml

This file sets the PWGAN generator and discriminator paramters.

## MelGAN.yaml

This file sets the PWGAN generator and discriminator paramters.


# Generate pattern

## Command
```
python Pattern_Generate.py [parameters]
```

## Parameters

At least, one or more of datasets must be used.

* -lj <path>
    * Set the path of LJSpeech. LJSpeech's patterns are generated.
* -bc2013 <path>
    * Set the path of Blizzard Challenge 2013. Blizzard Challenge 2013's patterns are generated.    
* -cmua <path>
    * Set the path of CMU arctic. CMU arctic's patterns are generated.
* -vctk <path>
    * Set the path of VCTK. VCTK's patterns are generated.
* -libri <path>
    * Set the path of LibriTTS. LibriTTS's patterns are generated.
* -text
    * Set whether the text information save or not.
    * This is for other model.
    * If you want to use for other models, the `Token_Path` parameter must be set in [Commons.yaml](./Hyper_Parameters/Commons.yaml)
* -eval
    * Set the evaluation pattern ratio.
    * Default is `0.001`.
* -mw
    * The number of threads used to create the pattern
    * Default is `10`.

# Run

## Modyfy inference file path while training for verification.

* Inference_Wav_for_Training.txt
    * Wav file paths which are used to evaluate while training.

## Command
```
python Train.py -s <int>
```

* `-s <int>`
    * The resume step parameter.
    * Default is 0.


# Results

* Please refere the demo site:
    * https://codejin.github.io/GAN_based_Vocoders_Demo

# Trained checkpoints

| Generator | Discriminator | Sample rate | Batch | Steps  | Checkpoint                                                                                 |
|-----------|---------------|-------------|-------|--------|--------------------------------------------------------------------------------------------|
| PWGAN     | PWGAN         | 16000       | 6     | 400000 | [Link](https://drive.google.com/file/d/1oRR5qgxbiu8C80YQr44KhFJVbJsJzbQs/view?usp=sharing) |
| PWGAN     | PWGAN         | 24000       | 6     | 400000 | [Link](https://drive.google.com/file/d/1DzQWYdDQSo3Dv_AA4aHIVLyRaUsqo-kE/view?usp=sharing) |
| PWGAN     | MelGAN        | 16000       | 6     | 349000 | [Link](https://drive.google.com/file/d/1Me8escBPc0Au4qbBFF_S5WKdZasAXYvH/view?usp=sharing) |


# Future works

1. Adding additional loss terms

2. Adding other GAN based vocoders
