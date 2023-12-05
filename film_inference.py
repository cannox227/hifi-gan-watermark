import os
import torch
from models import UNet, ConvDecoder
import numpy as np
import torch
import argparse
from inference import load_checkpoint, scan_checkpoint
from meldataset import load_wav, MAX_WAV_VALUE
from scipy.io.wavfile import write
from utils import get_audio_padded
from torch_pesq import PesqLoss
from meldataset import mel_spectrogram
import csv
# from torchmetrics.audio import PerceptualEvaluationSpeechQuality

def sort_key(file_name):
    # Split the file name using '_' and extract the number part
    return int(file_name.split('_')[2].split('.')[0])

def replace_positives(tensor):
    return torch.where(tensor > 0, torch.tensor(1), torch.tensor(0))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--input_wavs_dir', default='LJSpeech-1.1/wavs')
    parser.add_argument('--output_dir', default='film_generated_files')
    parser.add_argument('--input_validation_file', default='LJSpeech-1.1/validation.txt') 
    a = parser.parse_args()

    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    audio_channels = 1
    audio_channels_upsample = 256
    fing_size = 3
    encoder = UNet(n_class=audio_channels, fing_size=fing_size).to(device)
    decoder = ConvDecoder(input_size=audio_channels_upsample, output_size=fing_size).to(device)

    weights = sorted([f for f in os.listdir(a.checkpoint_file) if f.endswith('.pt')], reverse=True, key=sort_key)
    weights = weights[:2] # 0 is encoder, 1 is decoder
    state_dict_e = load_checkpoint(a.checkpoint_file+'/'+weights[0], device)
    encoder.load_state_dict(state_dict_e)
    state_dict_d = load_checkpoint(a.checkpoint_file+'/'+weights[1], device)
    decoder.load_state_dict(state_dict_d)

    # os.makedirs(a.output_dir, exist_ok=True)
    # filelist = os.listdir(a.input_wavs_dir)
    validation_files = []
    results = []
    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                            for x in fi.read().split('\n') if len(x) > 0]
    filelist = validation_files
    print(filelist)
    pesq_lib = PesqLoss(0.5, sample_rate=22050).to(device)
    # pesq_metric = PerceptualEvaluationSpeechQuality(22050)

    
    with torch.no_grad():
        for i, filename in enumerate(filelist):
            wav, sr = get_audio_padded(filename)
            wav = wav[:8192*20]
            # print(f"Audio shape: {wav.shape}")
            wav = wav / MAX_WAV_VALUE
            wav = torch.FloatTensor(wav).to(device)
            audio = wav.unsqueeze(0).unsqueeze(1).to(device)
            # print(f"Audio shape after being converted to tensor: {audio.shape}")
            random_fingerprint = torch.bernoulli(torch.full((1, fing_size), fill_value=0.5)).to(device)
            # print(f"Random fingerprint: {random_fingerprint}")
            audio_with_fingerprint = encoder(audio, random_fingerprint)
            fingerprint_hat = (decoder(audio_with_fingerprint))
            fingerprint_hat = replace_positives(fingerprint_hat)
            # print(f"Predicted fingerprint: {fingerprint_hat}")
            
            # write to file
            audio = audio_with_fingerprint.squeeze()
            audio = audio * MAX_WAV_VALUE
            # audio = audio.cpu().numpy().astype('int16')

           
            wav = wav.unsqueeze(0)
            audio = audio.unsqueeze(0)
            # print(f"Shapes : {audio.shape}, {wav.shape}")
            # print(f"Wav : {type(wav)}, audio {type(audio)}")
            mos = pesq_lib.mos(wav, audio)
            accuracy = torch.sum(random_fingerprint == fingerprint_hat)/fing_size
            # print("mos: ", mos)
            mel_original =  mel_spectrogram(wav.squeeze(1),1024, 80, 22050, 256, 1024,
                                        0, 8000)
            mel_hat = mel_spectrogram(audio.squeeze(1),1024, 80, 22050, 256, 1024,
                                        0, 8000)
            mel_loss = torch.nn.functional.l1_loss(mel_original, mel_hat)



            accuracy = torch.sum(random_fingerprint == fingerprint_hat)/fing_size
            results.append([mel_loss.item(), accuracy.item(), mos.item()])

            # If you want to generate audio uncomment the following lines
            # print("Audio shape before being written to file: ", audio.shape)
            # output_file = os.path.join(a.output_dir, os.path.splitext(filename)[0] + '_generated.wav')
            # write(output_file, sr, audio)

    csv_file_path = a.checkpoint_file + '/results.csv'
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Mel Loss", "Accuracy", "PESQ"])
        writer.writerows(results)
if __name__ == '__main__':
    main()