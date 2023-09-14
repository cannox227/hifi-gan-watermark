from models import *
import torch

def main():
    batch_size = 1
    channels = 512
    time = 1024
    x = torch.randn(batch_size, channels, time)
    w_channels = 256
    res = ResBlock1(h=1, channels=512, kernel_size=3, dilation=(1, 3, 5), film_channels = w_channels) 
    w = torch.randn(batch_size, w_channels, 1)
    y = res(x, w)
    print(y.shape)

if __name__ == '__main__':   
   main()