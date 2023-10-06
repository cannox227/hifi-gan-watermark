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

class SimpleDecoder(nn.Module):
    def __init__(self, input_size=128*128, output_size=128):
        super(SimpleDecoder, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)  # Adjust the hidden layer size as needed
        self.fc2 = nn.Linear(256, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Flatten the input tensor
        x = x.flatten()
        # Apply fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        
        # Apply sigmoid activation to output
        x = self.sigmoid(x)
        x  = (x >= 0.5).float()
        return x
    
def main():
    batch_size = 10
    channels = 512
    time = 1
    w_channels = 256
    mel_spectogram_channels = 80

    config_file = "config_custom.json"
    generator = get_generator_from_config(config_file)

    # input data
    # fake_tensor = torch.zeros(size=(128,128))
    bernoulli = BernoulliFingerprintEncoder(batch_size=1, probability=0.5)
    # fingerprint_encoded = bernoulli().squeeze(2)
    # fingerprint_original = bernoulli.get_original_fingerprint()
    # t_sum = (fake_tensor+fingerprint_original)

    loss_f = nn.MSELoss()

    #fingerprint = decoder(waveform)
    #print(fingerprint.requires_grad)
    #print("Decoder output shape: ",fingerprint.shape)
    
    # accuracy = (fingerprint == generator.original_fingerprint).float().mean()
    # print(f"Accuracy: {accuracy}")
    decoder = SimpleDecoder(input_size=128*128, output_size=128)

    epochs = 10000
    learning_rate = 0.001
    optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        decoder.train()
        
        fake_tensor = torch.zeros(size=(128,128), requires_grad=True)
        bernoulli()
        fingerprint_original = bernoulli.get_original_fingerprint()
        fingerprint_original.requires_grad_(True)
        t_sum = (fake_tensor+fingerprint_original) 
        fingerprint_hat = decoder(t_sum).unsqueeze(0)
        loss = loss_f(fingerprint_hat, fingerprint_original)

        optimizer.zero_grad()
        loss.require_grad = True
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

if __name__ == '__main__':   
   main()