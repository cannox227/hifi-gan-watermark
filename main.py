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
    time = 2
    w_channels = 256
    mel_spectogram_channels = 80

    config_file = "config_custom.json"
    generator = get_generator_from_config(config_file)
    
    # input data
    fake_tensor = torch.zeros(batch_size, mel_spectogram_channels, time, requires_grad=True) 
    #fake_tensor = torch.randn(batch_size, mel_spectogram_channels, time, requires_grad=True)
    
    #waveform computed for the first time in order to pass a correct shape to the decoder
    waveform = generator(fake_tensor)
    print(f"Waveform shape {waveform.shape}")
    decoder = FingerprintDecoder(input_channels=waveform.shape[2], hidden_layers=512)
    
    loss_f = nn.MSELoss()

    #fingerprint = decoder(waveform)
    #print(fingerprint.requires_grad)
    #print("Decoder output shape: ",fingerprint.shape)
    
    # accuracy = (fingerprint == generator.original_fingerprint).float().mean()
    # print(f"Accuracy: {accuracy}")

    epochs = 10000
    learning_rate = 0.001
    optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        decoder.train()
        generator.eval()
        with torch.no_grad():
            waveform = generator(fake_tensor)
    
        fingerprint = decoder(waveform)
        loss = loss_f(fingerprint, generator.original_fingerprint)

        optimizer.zero_grad()
        loss.require_grad = True
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

if __name__ == '__main__':   
   main()