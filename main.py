from models import *
from models import Generator
import torch
from utils import *
import json
from env import AttrDict
from torchsummary import summary
from meldataset import *
from scipy.io.wavfile import write

def set_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    return device

def get_generator_from_config(config_file, device):
    with open(config_file, "r") as config_file:
        config = json.load(config_file)
        params = AttrDict(config)

    generator = Generator(h=params).to(device)
    generator.load_state_dict(load_checkpoint(params.checkpoint_file, device))
    return generator

def get_mel(x, h):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)

def main():

    h = AttrDict()
    h.n_fft = 1024
    h.num_mels = 80
    h.sampling_rate = 22050
    h.hop_size = 256
    h.win_size = 1024
    h.fmin = 0
    h.fmax = 8000

    batch_size = 10
    channels = 512
    time = 2
    w_channels = 256
    mel_spectogram_channels = 80
    fingerprint_size = 128
    config_file = "config_custom.json"
    device = set_device()
    generator = get_generator_from_config(config_file, device)

    
    # input data -> mel spectogram
    #fake_tensor = torch.zeros(batch_size, mel_spectogram_channels, time, requires_grad=True) 
    PATH = 'experiments/LJ021-0165.wav'
    audio_padded, sr_padded = get_audio_padded(PATH)
    audio_padded = audio_padded / MAX_WAV_VALUE
    audio_padded = torch.FloatTensor(audio_padded).to(device)
    print("\n>MAIN: Audio padded shape: ", audio_padded.shape, audio_padded.dtype)
    mel_audio = get_mel(audio_padded.unsqueeze(0), h)
    print("\n>MAIN: Mel spectogram shape: ", mel_audio.shape, mel_audio.dtype)
    #fake_tensor = torch.randn(batch_size, mel_spectogram_channels, time, requires_grad=True)
    
    #waveform computed for the first time in order to pass a correct shape to the decoder
    waveform = generator(mel_audio)
    print(f"Generated Waveform shape {waveform.shape} {waveform.dtype}")
    # cfe = ConvFeatExtractor(use_spectral_norm=False)
    # feats = cfe(waveform)
    # print("Conv extractor shape: ", feats.shape)
    ad = AttentiveDecoder(input_dim=waveform.shape[2], output_dim=fingerprint_size)
    out = ad(waveform)
    print("Attentive decoder shape: ", out.shape)
    print("Out: ", out)

    # FOR GENERATING THE AUDIO SHAPE
    print(f"Waveform shape {waveform.shape} {waveform.dtype}")
    audio = waveform.squeeze()
    audio = audio * MAX_WAV_VALUE
    audio = audio.detach().numpy().astype('int16')
    # print("Audio shape before being written to file: ", audio.shape)
    output_file = os.path.join('experiments/generated.wav')
    write(output_file, h.sampling_rate, audio)


    return  
    
    # Attentive decoder
    decoder = AttentiveDecoder(input_dim=waveform.shape[1], output_dim=fingerprint_size)
    decoder(waveform)

    return 
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