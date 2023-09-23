from models import *
from models import Generator
import torch
from utils import *
import json
from env import AttrDict
from torchsummary import summary
def get_generator_from_config(config_file):
    with open(config_file, "r") as config_file:
        config = json.load(config_file)
        params = AttrDict(config)

    generator = Generator(h=params)
    return generator

def main():
    batch_size = 10
    channels = 512
    time = 1
    w_channels = 256
    mel_spectogram_channels = 80
    config_file = "config_custom.json"
    generator = get_generator_from_config(config_file)
    fake_tensor = torch.zeros(batch_size, mel_spectogram_channels, time)
    waveform = generator(fake_tensor)
    print(f"Waveform shape {waveform.shape}")
    
    decoder = FingerprintDecoder(input_channels=waveform.shape[2], hidden_layers=512)
    fingerprint = decoder(waveform)
    print("Decoder output shape: ",fingerprint.shape)
    
    accuracy = (fingerprint == generator.original_fingerprint).float().mean()
    print(f"Accuracy: {accuracy}")

if __name__ == '__main__':   
   main()