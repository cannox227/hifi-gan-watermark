from models import *
from models import Generator, BottleNeck, BottleNeckConv
import torch
from utils import *
import json
from env import AttrDict
from torch.utils.tensorboard import SummaryWriter

# def get_generator_from_config(config_file):
#     with open(config_file, "r") as config_file:
#         config = json.load(config_file)
#         params = AttrDict(config)

#     generator = Generator(h=params)
#     return generator

# class SimpleDecoder(nn.Module):
#     def __init__(self, input_size=128*128, output_size=128):
#         super(SimpleDecoder, self).__init__()
#         self.fc1 = nn.Linear(input_size, 256)  # Adjust the hidden layer size as needed
#         self.fc2 = nn.Linear(256, output_size)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # Flatten the input tensor
#         x = x.flatten()
#         # Apply fully connected layers
#         x = self.fc1(x)
#         x = self.fc2(x)
        
#         # Apply sigmoid activation to output
#         x = self.sigmoid(x)
#         x  = (x >= 0.5).float()
#         return x
    

class BasicEncoder(nn.Module):
    def __init__(self, input_size = 8192, fingerprint_size = 32, output_size = 32):
        super(BasicEncoder, self).__init__()
        self.input_size = input_size # segment size
        self.fingerprint_size = fingerprint_size 
        self.linear_input_size = self.input_size+self.fingerprint_size
        self.output_size = output_size
        self.layers = nn.ModuleList([
            nn.Linear(in_features=self.linear_input_size, out_features=self.linear_input_size // 2),
            nn.Linear(in_features=(self.linear_input_size // 2)+self.fingerprint_size, out_features=self.output_size)
        ])

    def forward(self, audio, fingerprint):
        #print()
        audio = audio.squeeze(1) # removing c=1 dimension
        #print(f"audio shape before concat ", audio.shape)
        audio_concat = torch.concat((audio, fingerprint), dim=1)
        #print(f"audio shape after concat ", audio_concat.shape)
        out = torch.relu(self.layers[0](audio_concat))
        #print(f"Shape after first layer ", out.shape)
        out = torch.concat((out, fingerprint), dim=1)
        #print(f"audio shape after second concat ", out.shape)
        out = self.layers[1](out)
        #print(f"Shape after second layer ", out.shape)
        return out


class BasicDecoder(nn.Module):
    def __init__(self, input_size = 4096, output_size = 8000):
        super(BasicDecoder, self).__init__()
        self.input_size = input_size # segment size
        self.output_size = output_size
        self.droput = nn.Dropout(p=0.25)
        self.layers = nn.ModuleList([
            nn.Linear(in_features=self.input_size, out_features=self.input_size * 2),
            nn.Linear(in_features=self.input_size * 2, out_features=self.input_size * 4),
            nn.Linear(in_features=self.input_size * 4, out_features=output_size), 
        ]) 

    def forward(self, x):
        #print(f"Decoder input shape ", x.shape)
        x = torch.relu(self.layers[0](x))
        x = self.droput(x)
        #print(f"Decoder out 1 shape ", x.shape)
        x = torch.relu(self.layers[1](x))
        x = self.droput(x)
        #print(f"Decoder out 2 shape ", x.shape)
        x = self.layers[2](x)
        #print(f"Decoder out 3 shape ", x.shape)
        # x = self.layers[1](x)
        #print(f"decoded x ", x)
        return x

class EncoderDecoder(nn.Module):
    def __init__(self, input_size = 4096, fingerprint_size = 32, embed_size = 16):
        super(EncoderDecoder, self).__init__()
        self.input_size = input_size # segment size
        self.fingerprint_size = fingerprint_size
    
        self.encoder = BasicEncoder(input_size=input_size, output_size=embed_size, fingerprint_size=fingerprint_size)
        self.decoder = BasicDecoder(input_size=embed_size, output_size=input_size) 

        self.bottleneck = nn.ModuleList([
            nn.Linear(in_features=input_size, out_features=input_size//2),
            nn.Linear(in_features=input_size // 2, out_features=fingerprint_size)
        ])

        
    def forward(self, x, fingerprint):
        x = self.encoder.forward(x, fingerprint)
        # print("fingerprint ", fingerprint)
        # print("encoded x ", x)
        x = self.decoder.forward(x)
        # print("decoded x ", x)
        # x = torch.relu(self.bottleneck[0](x))
        # #print("x shape after first bottleneck ", x.shape)
        # x = self.bottleneck[1](x)
        # #print("x after last layer no sigmoid ", x)
        # x = torch.sigmoid(x)
        #print("x shape after first bottleneck ", x.shape)
        #print("Final output ", x)
        return x

def main():
#     batch_size = 10
#     channels = 512
#     time = 1
#     w_channels = 256
#     mel_spectogram_channels = 80

#     config_file = "config_custom.json"
#     generator = get_generator_from_config(config_file)

#     # input data
#     # fake_tensor = torch.zeros(size=(128,128))
#     bernoulli = BernoulliFingerprintEncoder(batch_size=1, probability=0.5)
#     # fingerprint_encoded = bernoulli().squeeze(2)
#     # fingerprint_original = bernoulli.get_original_fingerprint()
#     # t_sum = (fake_tensor+fingerprint_original)

#     loss_f = nn.MSELoss()

#     #fingerprint = decoder(waveform)
#     #print(fingerprint.requires_grad)
#     #print("Decoder output shape: ",fingerprint.shape)
    
#     # accuracy = (fingerprint == generator.original_fingerprint).float().mean()
#     # print(f"Accuracy: {accuracy}")
#     decoder = SimpleDecoder(input_size=128*128, output_size=128)

#     epochs = 10000
#     learning_rate = 0.001
#     optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)


    ##########
    fingerprint_size = 2 
    embed_size = 512
    batch_size = 16 
    channels = 1
    time = 100
    epochs = 100000
    MAX_WAV_VALUE = 32768.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_tensor = torch.randint(low=-32768, high=32767, size=(batch_size, channels, time)).float().to(device) / MAX_WAV_VALUE
    fingerprint = torch.bernoulli(torch.full((batch_size, fingerprint_size), fill_value=0.5)).to(device)
    print(fingerprint.shape)
    print(input_tensor.shape)
    print(input_tensor)
    # encoder = BasicEncoder(input_size=time, fingerprint_size=fingerprint_size, output_size=output_size).to(device)
    # decoder = BasicDecoder(input_size=output_size, output_size=fingerprint_size).to(device)
    encdec = EncoderDecoder(input_size=time, fingerprint_size=fingerprint_size, embed_size=embed_size).to(device)
    #bottleneck = BottleNeck(input_size=time, output_size=fingerprint_size).to(device)
    bottleneck = BottleNeckConv(input_size=time, output_size=fingerprint_size).to(device)
    print("Bottleneck structure\n",bottleneck)
    # out = encoder(input_tensor, fingerprint)
    # print(out.shape)
    # print(out)
    fingerprint = torch.bernoulli(torch.full((batch_size, fingerprint_size), fill_value=0.5)).to(device)#.expand(batch_size, fingerprint_size).to(device)
    optim_encdec = torch.optim.AdamW(encdec.parameters(), lr= 0.0001, betas=[0.8, 0.99])
    optim_bottleneck = torch.optim.AdamW(bottleneck.parameters(), lr= 0.0001, betas=[0.8, 0.99])
    loss_f = nn.BCELoss()
    sigmoid = nn.Sigmoid()
    encdec.train()
    bottleneck.train()
    writer = SummaryWriter('experiments/logs')

    tensors = []
    for i in range(batch_size):
        tens = (torch.full(size=(channels, time), fill_value=i).to(device).float())
        tensors.append(tens)
    
    input_t = torch.stack(tensors).to(device)


    print("stacked tensor: ",input_t)
    for epoch in range(epochs):

        fingerprint = torch.bernoulli(torch.full((batch_size, fingerprint_size), fill_value=0.5)).to(device)#.expand(batch_size, fingerprint_size).to(device)
        # input_tensor = torch.randint(low=-32768, high=32767, size=(batch_size, channels, time)).float().to(device)  / MAX_WAV_VALUE
       
        # x_hat = encoder(input_tensor, fingerprint)
        # fing_hat = decoder(x_hat)

        encoded_audio = encdec(input_tensor, fingerprint).unsqueeze(1)
        #encoded_audio = input_t
        #print(encoded_audio.shape)
        fing_hat = bottleneck(encoded_audio)
        #print(fing_hat)
        #print(fing_hat[0], fing_hat[1])
        #print(f"Fing hat shape ", fing_hat.shape)
        #print(f"FINGERPRINT AFTER DECODER: ", fing_hat)
        # print(f"Orignal fingerpirnt: ", fingerprint) 
        # print(f"Orignal fingerpirnt shape: ", fingerprint.shape)
        #loss = torch.mean(torch.abs(fing_hat-fingerprint))
        loss = loss_f(fing_hat, fingerprint)
        #loss2 = loss_f2(fing_hat, fingerprint)
        #torch.mean(torch.binary_cross_entropy_with_logits(input=fing_hat, target=fingerprint))
        print(loss)
        #print(loss2)
        optim_encdec.zero_grad()
        optim_bottleneck.zero_grad()
        loss.backward()
        #loss2.backward()
        optim_encdec.step()
        optim_bottleneck.step()


        if (epoch + 1) % 100 == 0:
            #print(f"Fingerprint hat: {fing_hat}")
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}') 
            writer.add_scalar('training/decoder_error', loss, epoch)
    writer.close()
if __name__ == '__main__':   
   main()