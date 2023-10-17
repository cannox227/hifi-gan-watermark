from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from models import Generator, AttentiveDecoder
from utils import get_audio_padded
import numpy as np

h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    filelist = os.listdir(a.input_wavs_dir)

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        for i, filname in enumerate(filelist):
            # wav, sr = load_wav(os.path.join(a.input_wavs_dir, filname))
            # print(wav.shape, type(wav))

            # Read wav file with torchaudio and apply padding
            wav, sr = get_audio_padded(os.path.join(a.input_wavs_dir, filname))
            print(wav.shape)
            wav = wav / MAX_WAV_VALUE
            print("wav shape: ", wav.shape)
            wav = torch.FloatTensor(wav).to(device)
            x = get_mel(wav.unsqueeze(0))
            y_g_hat = generator(x)
            print("Generated audio shape: ", y_g_hat.shape)

            # Attentive decoder test
            fingerprint_size = 128 
            ad = AttentiveDecoder(input_dim=y_g_hat.shape[2], output_dim=fingerprint_size)
            out = ad(y_g_hat)
            print("Attentive decoder shape: ", out.shape)
            print("Fingerprint tensor: ", out)
            fing_arr = out.squeeze().cpu().numpy().astype('int16')
            fing_conv = decimal_value = int(''.join(map(str, fing_arr.tolist())), 2)
            print("Fingerprint converted: ", fing_conv)

            # Convert generated audio to wav and write on disk
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')
            print("Audio shape before being written to file: ", audio.shape)
            output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '_generated.wav')
            write(output_file, h.sampling_rate, audio)
            print(output_file)

            test_wav, sr = load_wav(output_file)
            wav = torch.FloatTensor(wav).to(device)
            print("Wav file after reading it with torchaudio: ", wav.shape)
            x = get_mel(wav.unsqueeze(0)) 
            print("Mel spectrogram shape: ", x.shape)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='test_files')
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()

