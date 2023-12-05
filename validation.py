import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict, build_env
from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
from models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, \
    BottleNeckConv, AttentiveDecoder, BottleNeck, ConvDecoder,\
feature_loss, generator_loss,\
    discriminator_loss
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint
from torch_pesq import PesqLoss

torch.backends.cudnn.benchmark = True

def replace_positives(tensor):
    return torch.where(tensor > 0, torch.tensor(1), torch.tensor(0))

def train(rank, a, h):
    print(dir(a))
    print(a)
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    generator = Generator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    wm_decoder =  ConvDecoder(input_size=h.film_channels, output_size=h.fingerprint_size).to(device)
    pesq_lib = PesqLoss(0.5, sample_rate=22050).to(device)
    #BottleNeckConv(input_size=h.segment_size, output_size=generator.bernoulli.fingerprint_size).to(device)
    #BottleNeck(input_size=h.segment_size, output_size=generator.bernoulli.fingerprint_size).to(device)
    # AttentiveDecoder(input_dim=h.segment_size,output_dim=generator.bernoulli.fingerprint_size).to(device)   
    decoder_loss = torch.nn.BCELoss().to(device)

    if rank == 0:
        print(generator)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')
        # TODO: add checkpoint for wm_decoder
        cp_wm = scan_checkpoint(a.checkpoint_path, 'wm_')


    steps = 0
    if cp_g is None or cp_do is None or cp_wm is None:
        state_dict_do = None
        state_dict_wm = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        state_dict_wm = load_checkpoint(cp_wm, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        wm_decoder.load_state_dict(state_dict_wm['wm'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']
        #TODO: load wm decoder dict

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)
        wm_decoder = DistributedDataParallel(wm_decoder, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_wm = torch.optim.AdamW(wm_decoder.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])


    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])
    
    if state_dict_wm is not None:
        optim_wm.load_state_dict(state_dict_wm['optim_wm'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_wm = torch.optim.lr_scheduler.ExponentialLR(optim_wm, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(a)

    trainset = MelDataset(training_filelist, h.segment_size, h.n_fft, h.num_mels,
                          h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                          shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax_for_loss, device=device,
                          fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir)

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)
    ## TODO: Augmentation 
    if rank == 0:
        validset = MelDataset(validation_filelist, h.segment_size, h.n_fft, h.num_mels,
                              h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, False, False, n_cache_reuse=0,
                              fmax_loss=h.fmax_for_loss, device=device, fine_tuning=a.fine_tuning,
                              base_mels_path=a.input_mels_dir)
        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)

        ## TODO: Batch valset and valloader
        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    generator.eval()
    mpd.eval()
    msd.eval()
    wm_decoder.eval()
    

    for i, batch in enumerate(validation_loader):
        if rank == 0:
            start_b = time.time()
        x, y, _, y_mel = batch
        x = torch.autograd.Variable(x.to(device, non_blocking=True))
        y = torch.autograd.Variable(y.to(device, non_blocking=True))
        y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
        y = y.unsqueeze(1)

        y_g_hat = generator(x)
        y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size,
                                        h.fmin, h.fmax_for_loss)

        # L1 Mel-Spectrogram Loss
        loss_mel = F.l1_loss(y_mel, y_g_hat_mel) 
        
        fp_hat = wm_decoder(y_g_hat)
        fp_hat = replace_positives(fp_hat)
        fp_true = generator.bernoulli.get_original_fingerprint()
        # print(f"FP TRUE: {fp_true}")
        # print(f"FP HAT: {fp_hat}")
        accuracy = torch.sum(fp_hat == fp_true)/h.fingerprint_size
        diff = y.shape[2] - y_g_hat.shape[2]
        print(diff)
        y_g_hat = torch.nn.functional.pad(y_g_hat, (0, diff))
        print(y.shape, y_g_hat.shape)
        pesq = pesq_lib.mos(y.squeeze(1), y_g_hat.squeeze(1))
        pesq = (pesq.item())

        
        print('Steps: {:d}, Mel-Spec. Error: {:4.3f}, Decoder-Accuracy: {:4.3f}, PESQ: {:4.3f}'.
              format(steps, loss_mel, accuracy, pesq))

        steps += 1

def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_wavs_dir', default='LJSpeech-1.1/wavs')
    parser.add_argument('--input_mels_dir', default='ft_dataset')
    parser.add_argument('--input_training_file', default='LJSpeech-1.1/training.txt')
    parser.add_argument('--input_validation_file', default='LJSpeech-1.1/validation.txt')
    parser.add_argument('--checkpoint_path', default='cp_hifigan')
    parser.add_argument('--config', default='')
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()
