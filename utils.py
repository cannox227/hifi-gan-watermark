import glob
import os
import matplotlib
import torch
from torch.nn.utils import weight_norm
matplotlib.use("Agg")
import matplotlib.pylab as plt
import librosa
import numpy as np
from meldataset import load_wav

MAX_AUDIO_LEN_S = 10.20

def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

def get_mfcc_features(audio_data : np.ndarray, n_mfcc=20, sr=22050):
    mfcc = librosa.feature.mfcc(y=audio_data.astype(np.float16), n_mfcc=n_mfcc, sr=sr)
    return mfcc

def get_mfcc_features_padded(audio_data : np.ndarray, n_mfcc=20, sr=22050):
    # Because S parameter is None, Mel spectorgram is generated 
    # with hop_length = 512 and n_fft = 2048 
    # mffc.shape = (mfcc_channels, audio_len / hop_length)
    mfcc = librosa.feature.mfcc(y=audio_data, n_mfcc=n_mfcc, sr=sr)
    # Padding is added post audio feature retrieval
    # Given an audio with sr=22050 Hz with max lenght of 10.30 seconds 
    # we get sr * duration = 227115 samples
    MAX_N_SAMPLES = sr * MAX_AUDIO_LEN_S
    MAX_LEN = round(MAX_N_SAMPLES / 512)

    padding_size = MAX_LEN - mfcc.shape[1]
    print(padding_size)
    padding = np.zeros((n_mfcc, padding_size), dtype=np.int16)
    mfcc = np.concatenate((mfcc, padding), axis=1, dtype=np.int16)
    
    return mfcc

def get_padded_audio_len(sample_rate: int):
    # Return max audio length after padding 
    return round(sample_rate * MAX_AUDIO_LEN_S)

def get_audio_padded(file_path):
    audio_data,fs  = load_wav(file_path)
    #print("fs", fs)
    #print(type(audio_data), audio_data.dtype)
    audio_len = get_padded_audio_len(fs)
    padding_size = int(audio_len - audio_data.shape[0])
    padding = np.zeros(padding_size, dtype=np.int16)
    audio_data = np.concatenate((audio_data, padding), axis=0, dtype=np.int16)
    return audio_data, fs

def save_mel_spectrogram(audio, sr, filepath):
    audio = audio.astype(np.float16)
    # Create a spectrogram
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)

    # Convert to dB scale (log scale)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    plt.imshow(spectrogram_db, origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.savefig(filepath+".png")

def save_mfcc_plot(mfcc_vals, filepath):
    plt.figure(figsize=(10, 6))
    plt.imshow(mfcc_vals, origin='lower', aspect='auto')
    plt.colorbar(format='%+2.0f dB')
    plt.yticks(np.arange(0, mfcc_vals.shape[0], 1))
    plt.title('MFCC')
    plt.savefig(filepath+".png")

