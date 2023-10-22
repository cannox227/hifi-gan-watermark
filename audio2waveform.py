import librosa
import torch
import numpy as np
import scipy
from scipy.io.wavfile import write
from meldataset import *
from utils import *

N_MFCC_FEATS = 20
# PATHS = ['experiments/LJ001-0001.wav',
#          'experiments/LJ002-0321.wav',
#          'experiments/LJ021-0165.wav']

PATHS = ['experiments/LJ001-0001.wav','generated_files/LJ001-0001_generated.wav']
# for i, PATH in enumerate(PATHS):
#     audio, sr = load_wav(PATH)
    #print("Original audio length: ", audio.shape, audio.dtype)
    #audio_padded, sr_padded = get_audio_padded(PATH)
    #print("Padded audio length: ", audio_padded.shape, audio_padded.dtype)
    #feat = get_mfcc_features(audio, n_mfcc=N_MFCC_FEATS, sr=sr)
    #print("MFCC features before padding",feat.shape)
    #feat_padded = get_mfcc_features(audio_padded, n_mfcc=N_MFCC_FEATS, sr=sr_padded)
    #print("MFCC features after padding", feat_padded.shape)
    #filename = PATH.replace('experiments/', '')
    #compute_and_save_mel_spectrogram(audio, sr, f'experiments/MEL_OG_{i}')
    #save_mel_spectrogram(audio_padded, sr, 'experiments/MEL_PADDED_'+filename)
    #save_mfcc_plot(feat, 'experiments/MFCC_'+PATH.replace('experiments/', ''))
    #save_mfcc_plot(feat_padded, 'experiments/MFCC_padded'+PATH.replace('experiments/', ''))
mels = []
for i, a in enumerate(PATHS):
    if i == 0:
        audio, sr = get_audio_padded(a)
    else:
        audio, sr = load_wav(a)
    compute_and_save_mel_spectrogram(audio, sr, f'experiments/test/MEL_OG_{i}')
    audio = torch.FloatTensor(audio)
    audio = audio.unsqueeze(0)
    mel_no_finger = mel_spectrogram(y=audio, n_fft=1024, num_mels=80, sampling_rate=22050, hop_size=256, win_size = 1024, fmin = 0, fmax = 8000, center=False)
    mels.append(mel_no_finger)
    save_mel_spectrogram(mel=mel_no_finger.squeeze(0), filepath=f'experiments/test/MEL_NO_FINGER_{i}')
    fing = torch.randint_like(mel_no_finger, high=20, low=0)
    mel_wm = fing + mel_no_finger
    save_mel_spectrogram(mel=mel_wm.squeeze(0), filepath=f'experiments/test/MEL_W_FINGER_{i}')

den = torch.nn.functional.mse_loss(input=mels[1], target=mels[0])
num = torch.max(mels[1].squeeze(0))**2
print("num: ", num, " den ", den)
psnr = 10*torch.log10(num/den)
print("PSNR: ",psnr)
