import librosa
import torch
import numpy as np
import scipy
from scipy.io.wavfile import write

from utils import *

N_MFCC_FEATS = 20
PATHS = {'experiments/LJ001-0001.wav',
         'experiments/LJ002-0321.wav',
         'experiments/LJ021-0165.wav'}

for PATH in PATHS:
    audio, sr = load_wav(PATH)
    print("Original audio length: ", audio.shape, audio.dtype)
    audio_padded, sr_padded = get_audio_padded(PATH)
    print("Padded audio length: ", audio_padded.shape, audio_padded.dtype)
    feat = get_mfcc_features(audio, n_mfcc=N_MFCC_FEATS, sr=sr)
    print("MFCC features before padding",feat.shape)
    feat_padded = get_mfcc_features(audio_padded, n_mfcc=N_MFCC_FEATS, sr=sr_padded)
    print("MFCC features after padding", feat_padded.shape)
    filename = PATH.replace('experiments/', '')
    save_mel_spectrogram(audio, sr, 'experiments/MEL_'+filename)
    save_mel_spectrogram(audio_padded, sr, 'experiments/MEL_PADDED_'+filename)
    save_mfcc_plot(feat, 'experiments/MFCC_'+PATH.replace('experiments/', ''))
    save_mfcc_plot(feat_padded, 'experiments/MFCC_padded'+PATH.replace('experiments/', ''))

