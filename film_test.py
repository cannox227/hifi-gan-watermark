import argparse
import json
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import weight_norm
import datetime
from env import AttrDict 
from meldataset import MelDataset, get_dataset_filelist
from torch.utils.data import DistributedSampler, DataLoader
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from utils import get_padding
MAX_WAV_VALUE = 32768.0
device =  'cuda' if torch.cuda.is_available() else 'cpu'

class Net(nn.Module):
    def __init__(self, fingeprint_size, embedding_size):
        super(Net, self).__init__()
        self.watermark_embed = nn.Linear(fingeprint_size, embedding_size)
        self.conv_pre_film = nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim*2, kernel_size=1, stride=1, padding=0)
        self.conv_post_film = nn.Conv1d(in_channels=embedding_dim, out_channels=1, kernel_size=1, stride=1, padding=0)
    def forward(self, watermark, x):
        watermark = self.watermark_embed(watermark)
        # print("pre watermark changing dim shape: ", watermark.shape)
        watermark = watermark.unsqueeze(2).expand(-1, -1, x.size(2))
        # print("after watermark changing dim shape: ", watermark.shape)

        a,b = torch.chunk(self.conv_pre_film(watermark), chunks=2, dim=1)
        output = x*a+b
        output = self.conv_post_film(output)
        return output
    

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        # self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(input_size, input_size // 2)
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(input_size // 2, input_size // 4)
        # self.fc3 = nn.Linear(input_size // 4, input_size // 8)
        # self.fc4 = nn.Linear(input_size // 8, input_size // 16)
        # self.fc5 = nn.Linear(input_size // 16, output_size
        self.input_size = input_size
        
        self.proj = weight_norm(nn.Conv1d(in_channels=1, out_channels=input_size, kernel_size=1, padding='same'))
        self.convs = nn.ModuleList([
        #weight_norm(nn.Conv1d(input_size, input_size//4, kernel_size=3, stride=1, dilation=1, padding='same')),
        weight_norm(nn.Conv1d(input_size, input_size//8, kernel_size=3, stride=1, dilation=2, padding=get_padding(3,2))),
        #weight_norm(nn.Conv1d(input_size//8, input_size//8, kernel_size=5, stride=1, dilation=1, padding='same')),
        weight_norm(nn.Conv1d(input_size//8, input_size//16, kernel_size=5, stride=1, dilation=4, padding=get_padding(5,4))),
        #weight_norm(nn.Conv1d(input_size//16, input_size//16, kernel_size=5, stride=1, dilation=1, padding='same')),
        weight_norm(nn.Conv1d(input_size//16, input_size//32, kernel_size=5, stride=1, dilation=8, padding=get_padding(5,8))),
        # weight_norm(nn.Conv1d(input_size//32, input_size//16, kernel_size=3, stride=1, padding='same')),
        # weight_norm(nn.Conv1d(input_size//16, input_size//32, kernel_size=3, stride=1, padding='same')),
        # weight_norm(nn.Conv1d(input_size//16, input_size//128, kernel_size=3, stride=1, padding='same')),
        #weight_norm(nn.Conv1d(input_size//32, input_size//32, kernel_size=5, stride=1, padding='same')),
        weight_norm(nn.Conv1d(input_size//32, input_size//64, kernel_size=3, stride=1, dilation=16, padding=get_padding(3,16))),
        weight_norm(nn.Conv1d(input_size//64, output_size, kernel_size=3, stride=1, padding='same'))
        ])
        self.leakyRELU = nn.LeakyReLU(0.01)
        self.avgPool = nn.AdaptiveAvgPool1d(output_size=16)

    def forward(self, x):
        # x = x.unsqueeze(0).unsqueeze(0)
        # # instead of interpolating just use a 1x1 conv
        # x = torch.nn.functional.interpolate(x, scale_factor=(1, self.input_size, 1), mode='nearest')
        # x = x.squeeze(0).squeeze(0)
        x = self.proj(x)
        for i, c in enumerate(self.convs):
            if i != len(self.convs)-1:
                x = self.leakyRELU(c(x))
            else:
                x = c(x)
        x = torch.mean(x, dim=-1).squeeze(-1)
        # x = self.flatten(x)
        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        # x = self.relu(x)
        # x = self.fc3(x)
        # x = self.relu(x)
        # x = self.fc4(x)
        # x = self.relu(x)
        # x = self.fc5(x)
        return x


# Example usage:
batch_size = 16
f_size = 3
channels = 256
time = 8000
audio_channels = 256 # Upsample channels
embedding_dim = 256
# Random input tensors for testing
z_tensor =  torch.bernoulli(torch.full((batch_size, f_size), fill_value=0.5)).to(device)
x_tensor = torch.randint(low=-32768, high=32767, size=(batch_size, channels, time)).float().to(device)  / MAX_WAV_VALUE

print(f"Watermark shape: {z_tensor.shape}\nInput shape: {x_tensor.shape}")
net = Net(f_size, embedding_dim).to(device)

out = net(z_tensor, x_tensor)
#mlp = MLP(audio_channels * time, f_size).to(device)
mlp = MLP(audio_channels, f_size).to(device)
dec = mlp(out)
print(out.shape)
print(dec.shape)

namefilm = f"REAL_AUDIO_14_{len(mlp.convs)}_Time_{time}_F_BIT_{f_size}_UPSAMPLE_{audio_channels}"
writer = SummaryWriter(f'experiments/film-logs/{namefilm}')

optim_dec = torch.optim.AdamW(mlp.parameters(), lr=0.0001, weight_decay=0.01)
criterion = nn.BCELoss().to(device)

epochs = 10000
print(mlp)

# Mel dataset
parser = argparse.ArgumentParser()
parser.add_argument('--group_name', default=None)
parser.add_argument('--input_wavs_dir', default='LJSpeech-1.1/wavs')
parser.add_argument('--input_mels_dir', default='ft_dataset')
parser.add_argument('--input_training_file', default='LJSpeech-1.1/training.txt')
parser.add_argument('--input_validation_file', default='LJSpeech-1.1/validation.txt')
parser.add_argument('--checkpoint_path', default='cp_hifigan')
parser.add_argument('--config', default='config_custom.json')
parser.add_argument('--training_epochs', default=3100, type=int)
parser.add_argument('--stdout_interval', default=5, type=int)
parser.add_argument('--checkpoint_interval', default=5000, type=int)
parser.add_argument('--summary_interval', default=100, type=int)
parser.add_argument('--validation_interval', default=1000, type=int)
parser.add_argument('--fine_tuning', default=False, type=bool)

a = parser.parse_args()
training_filelist, _ = get_dataset_filelist(a)
with open(a.config) as f:
        data = f.read()

json_config = json.loads(data)
h = AttrDict(json_config)
trainset = MelDataset(training_filelist, h.segment_size, h.n_fft, h.num_mels,
                          h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                          shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax_for_loss, device=device,
                          fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir)

torch.manual_seed(h.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(h.seed)
    h.num_gpus = torch.cuda.device_count()
    h.batch_size = int(h.batch_size / h.num_gpus)
    print('Batch size per GPU :', h.batch_size)
else:
    pass

train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)

net.eval()
mlp.train()
epochs = 50
for e in range(epochs):
    # z_tensor =  torch.bernoulli(torch.full((batch_size, f_size), fill_value=0.5)).to(device)
    # x_tensor = torch.randint(low=-32768, high=32767, size=(batch_size, channels, time)).float().to(device)  / MAX_WAV_VALUE

    # out = net(z_tensor, x_tensor)
    # fing = mlp(out)

    # loss = criterion(torch.sigmoid(fing), z_tensor)

    if h.num_gpus > 1:
            train_sampler.set_epoch(e)

    for i, batch in enumerate(train_loader):
        if i < 400:
            x, y, _, y_mel = batch
            y = y.unsqueeze(1).to(device)
            z_tensor =  torch.bernoulli(torch.full((batch_size, f_size), fill_value=0.5)).to(device)
            out = net(z_tensor, y)
            fing = mlp(out)

            loss = criterion(torch.sigmoid(fing), z_tensor)

            optim_dec.zero_grad()
            loss.backward()
            optim_dec.step()
        else: 
            break
        print(f"Epoch [{e}/{epochs}], Loss {loss.item()}")
    writer.add_scalar('log/mlp-film-decoder', loss, e)
    if (e +1)%10 == 0:
        print(f"Epoch [{e}/{epochs}], Loss {loss.item()}")
        writer.add_scalar('log/mlp-film-decoder', loss, e)
writer.close()
