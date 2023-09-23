from models import *
import torch
from utils import *

def main():
    batch_size = 1
    channels = 512
    time = 1024

    x = torch.randn(batch_size, channels, time)
    w_channels = 256
    res = ResBlock1(h=1, channels=512, kernel_size=3, dilation=(1, 3, 5), film_channels = w_channels) 
    w = torch.randn(batch_size, w_channels, 1)
    y = res(x, w)
    print(f"Image with watermark shape: {y.shape}")

    decoder = FingerprintDecoder(input_channels=y.shape[1], hidden_layers=512, time=y.shape[2])
    z = decoder(y)
    print(f"Fingerprint obtained {z.shape}")

    encoder = BernoulliFingerprintEncoder(probability=0.5)
    fingerprint_enc = encoder()
    
    print(f"\nFingerprint encoded {fingerprint_enc.shape}, \nEncoded val: {fingerprint_enc}")


if __name__ == '__main__':   
   main()