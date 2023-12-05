import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
#from torch.nn.utils.parametrizations import weight_norm
from utils import init_weights, get_padding

LRELU_SLOPE = 0.1


class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        # If film channels are specified in the configuration file
        # FILM is applied to the residual block
        if h.film_channels > 0: 
            #print("using film channels feature") 
            self.film_channels = h.film_channels
            self.conv_film = weight_norm(Conv1d(in_channels=self.film_channels, out_channels=channels*2, kernel_size=1, 
                                                stride=1, padding=0))
            self.conv_film.apply(init_weights)
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x, w):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            # Add the feature wise linear modulation
            if self.film_channels > 0:
                #print("Using film channels")
                a, b = torch.chunk(self.conv_film(w), 2, dim=1)
                xt = a * xt + b
                x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        # If film channels are specified in the configuration file
        # FILM is applied to the residual block
        self.film_channels = 0
        if h.film_channels > 0:
            #print("using film channels feature") 
            self.film_channels = h.film_channels
            self.conv_film = weight_norm(Conv1d(in_channels=self.film_channels, out_channels=channels*2, kernel_size=1, 
                                                stride=1, padding=0))
            self.conv_film.apply(init_weights)
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x, w):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
            # Add the feature wise linear modulation
            #print("x shape before FILM",x.shape)
            if self.film_channels > 0:
                #print("using film channels feature") 
                a, b = torch.chunk(self.conv_film(w), 2, dim=1)
                #print("a shape, b shape",a.shape,b.shape)
                xt = a * xt + b
                x = xt + x
            #print("x shape after FILM",x.shape)
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2
        self.film_channels = h.film_channels
        if self.film_channels > 0:
            self.original_fingerprint = None
            self.bernoulli = BernoulliFingerprintEncoder(probability=0.5,fingerprint_size=h.fingerprint_size) #TODO: add inferprint size
        self.ups = nn.ModuleList()

        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))
                
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        if self.film_channels > 0:
            self.fingerprint = self.bernoulli(x)
            self.original_fingerprint = self.bernoulli.get_original_fingerprint()
        else:
            self.fingerprint = None
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x) # ky[l] * 1 ConvTranspose
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x, self.fingerprint)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x, self.fingerprint)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        x.requires_grad_(True)
        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

        


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

class FingerprintDecoder(torch.nn.Module):
    def __init__(self, input_channels=256, hidden_layers=512):
        super(FingerprintDecoder, self).__init__()

        self.fingerprint_size = 128
        self.threshold = 0.5
        self.linears = nn.ModuleList([
            torch.nn.Linear(input_channels, hidden_layers),
            torch.nn.Linear(hidden_layers, hidden_layers // 2),
            torch.nn.Linear(hidden_layers // 2, self.fingerprint_size)
        ])
        
        self.lrelu = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = x.flatten(1)
        #print("x shape after flatten", x.shape)
        for l in self.linears:
            if l == self.linears[-1]:
                x = self.sigmoid(l(x))
            else:
                x = self.lrelu(l(x))
        x  = (x >= self.threshold).float()
        x.requires_grad_(True)
        return x
    
class BernoulliFingerprintEncoder(torch.nn.Module):
    def __init__(self, probability=0.5, fingerprint_size=8):
        super(BernoulliFingerprintEncoder, self).__init__()
        #self.bern_shape = (batch_size, 128)
        self.fingerprint_size = fingerprint_size
        self.hidden_size = self.fingerprint_size
        self.output_size = 256
        # output size should match the film_channels parameter
        self.original_fingerprint = None
        self.linears =  self.linears = nn.ModuleList([
            torch.nn.Linear(self.hidden_size, self.hidden_size*2),
            torch.nn.Linear(self.hidden_size*2, self.output_size)
        ]) 
        self.relu = torch.nn.ReLU()
        self.prob = probability

    def forward(self, x):
        # different fingerprint for each element in the batch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #self.original_fingerprint = torch.ones((x.shape[0], self.fingerprint_size)).to(device).requires_grad_(True)
        self.original_fingerprint = torch.bernoulli(torch.full((x.shape[0],self.fingerprint_size), self.prob)).to(device)
        x = self.relu(self.linears[0](self.original_fingerprint))
        x = self.linears[1](x)
        return x.unsqueeze(2)
    
    def get_original_fingerprint(self):
        return self.original_fingerprint
        
class SelfAttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttentionLayer, self).__init__()
        self.input_dim = input_dim
        
        # Define weight matrices for query, key, and value
        self.W_q = nn.Linear(input_dim, input_dim)
        self.W_k = nn.Linear(input_dim, input_dim)
        self.W_v = nn.Linear(input_dim, input_dim)
        
    def forward(self, x):
        # Compute query, key, and value
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        
        # Calculate scaled dot-product attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.input_dim, dtype=torch.float32))
        
        # Apply softmax to obtain attention weights
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        
        # Compute weighted sum using attention weights
        output = torch.matmul(attention_weights, v)
        
        return output

class ConvFeatExtractor(torch.nn.Module):
    def __init__(self, time_channel = 1, use_spectral_norm=False):
        super(ConvFeatExtractor, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.output_dim = 1024
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)), 
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)), 
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)), 
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, self.output_dim, 5, 1, padding=2)),
        ])
        # If input [1, 1, 224768]
        # conv time out will be the third dimension divided by 
        # the stride of each conv
        self.conv_time_out = time_channel / (1*2*2*4*4*1*1)
    def forward(self, x):
        for i, l in enumerate(self.convs):
            x = l(x)
            if i != len(self.convs)-1:
                x = F.leaky_relu(x, LRELU_SLOPE)   
            #print("x shape after all convs ",x.shape)         
        return x

class BottleNeck(nn.Module):
    def __init__(self, input_size = 8000, output_size = 32):
        super(BottleNeck, self).__init__()
        self.input_size = input_size
        self.outpur_size = output_size #fingerprint size
        self.dropout = nn.Dropout(p=0.25)
        self.bottlenecks = nn.ModuleList([
            nn.Linear(in_features=input_size, out_features=input_size//4),
            nn.Linear(in_features=input_size // 4, out_features=input_size // 16),
            nn.Linear(in_features=input_size // 16, out_features=input_size // 64),
            nn.Linear(in_features=input_size // 64, out_features=output_size)
        ])

    def forward(self, x):
        for i, bn in enumerate(self.bottlenecks):
            if i != len(self.bottlenecks)-1:
                x = torch.relu(bn(x))
                x = self.dropout(x)
        x = torch.sigmoid(self.bottlenecks[-1](x)) 
        return x

class BottleNeckConv(nn.Module):

    
    def __init__(self, input_size = 8000, output_size = 32):
        super(BottleNeckConv, self).__init__()
        self.input_size = input_size
        self.output_size = output_size #fingerprint size

        #kernerl, s, d
        k_s = 7
        d = 4
        channels = [256, 256, 256]#, 32, 64, 64, 256, 256, 512, 512, 1024, 1024]
        kernels = [1, 1, 1]#, 59, 67, 71, 79, 83, 89, 97, 97, 97]
        self.convs = nn.ModuleList()
        for i, (c, k) in enumerate(zip(channels, kernels)):
            if i == 0:
                self.convs.append(weight_norm(nn.Conv1d(in_channels=3, out_channels=c, kernel_size=k, dilation=d, padding='same')))
            else:
                self.convs.append(weight_norm(nn.Conv1d(in_channels=channels[i-1], out_channels=c, kernel_size=k, dilation=d, padding='same'))) 


        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.last_conv = weight_norm(nn.Conv1d(in_channels=channels[-1], out_channels=output_size, kernel_size=1, padding='same'))
    

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            x = self.relu(conv(x))

        x = torch.mean(x, dim=-1, keepdim=True) 
        
        x = torch.sigmoid(self.last_conv(x)) 
        x = x.squeeze(2)
        return x
        
class AttentiveDecoder(nn.Module):
    # Input dim = mcpp channels
    # Output dim = fingerprint size
    def __init__(self, input_dim = 1, output_dim = 128):
        super(AttentiveDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.threshold = 0.5

        self.feature_extractors = ConvFeatExtractor(time_channel=input_dim, use_spectral_norm=True)
        self.conv_time = int(self.feature_extractors.conv_time_out)
        self.attention = SelfAttentionLayer(input_dim=self.conv_time)

        self.bottlenecks = nn.ModuleList([
            nn.Linear(self.conv_time * 2, self.conv_time), #3512
            nn.Linear(self.conv_time, self.conv_time // 2), #1756
            nn.Linear(self.conv_time // 2, self.conv_time // 4),#878  
            nn.Linear(self.conv_time // 4, self.output_dim) # 128
        ])

        assert self.output_dim == 32, "Output dim must be 32!"

    def weighted_sd(self,inputs,attention_weights, mean):
        el_mat_prod = torch.mul(inputs,attention_weights)
        hadmard_prod = torch.mul(inputs,el_mat_prod)
        # TODO why variance is used instead of std?
        variance = torch.sum(hadmard_prod,1) - torch.mul(mean,mean)
        return variance
    
    def stat_attn_pool(self,inputs,attention_weights):
        el_mat_prod = torch.mul(inputs,attention_weights)
        # Do we want to compute the mean of each sample over all the available features
        mean = torch.mean(el_mat_prod,1)    #[1, 3512]
        variance = self.weighted_sd(inputs,attention_weights,mean)
        stat_pooling = torch.cat((mean,variance),1)
        return stat_pooling
    
    # What is loss baby don't hurt me, don't hurt me, no more
    def forward(self, y):#, y_hat):
        # R = real
        # G = generated
        y = self.feature_extractors(y)
        # Apply self attention
        y_att_weights = self.attention(y)
        # Apply stat pooling
        y = self.stat_attn_pool(y, y_att_weights)
        for i,mlp in enumerate(self.bottlenecks):
            y = mlp(y)
            if i != len(self.bottlenecks)-1:
                y = F.leaky_relu(y, LRELU_SLOPE)

        y.requires_grad_(True)
        return y
    
class ConvDecoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(ConvDecoder, self).__init__()
        self.input_size = input_size
        
        self.proj = weight_norm(nn.ConvTranspose1d(in_channels=1, out_channels=input_size, kernel_size=1))#, padding='same'))
        self.convs = nn.ModuleList([
        # weight_norm(nn.Conv1d(input_size, input_size//4, kernel_size=3, stride=1, dilation=1, padding='same')),
        weight_norm(nn.Conv1d(input_size, input_size//8, kernel_size=3, stride=1, dilation=2, padding=get_padding(3,2))),
        weight_norm(nn.Conv1d(input_size//8, input_size//8, kernel_size=5, stride=1, dilation=1, padding='same')),
        weight_norm(nn.Conv1d(input_size//8, input_size//16, kernel_size=5, stride=1, dilation=4, padding=get_padding(5,4))),
        # weight_norm(nn.Conv1d(input_size//16, input_size//16, kernel_size=5, stride=1, dilation=1, padding='same')),
        weight_norm(nn.Conv1d(input_size//16, input_size//32, kernel_size=5, stride=1, dilation=8, padding=get_padding(5,8))),
        # weight_norm(nn.Conv1d(input_size//32, input_size//16, kernel_size=3, stride=1, padding='same')),
        # weight_norm(nn.Conv1d(input_size//16, input_size//32, kernel_size=3, stride=1, padding='same')),
        # weight_norm(nn.Conv1d(input_size//16, input_size//128, kernel_size=3, stride=1, padding='same')),
        # weight_norm(nn.Conv1d(input_size//32, input_size//32, kernel_size=5, stride=1, padding='same')),
        weight_norm(nn.Conv1d(input_size//32, input_size//64, kernel_size=3, stride=1, dilation=16, padding=get_padding(3,16))),
        # weight_norm(nn.Conv1d(input_size//64, input_size//64, kernel_size=3, stride=1, padding='same')),
        weight_norm(nn.Conv1d(input_size//64, output_size, kernel_size=3, stride=1, padding='same'))
        ])
        self.leakyRELU = nn.LeakyReLU(0.01)
        self.avgPool = nn.AdaptiveAvgPool1d(output_size=16)

    def forward(self, x):
        x = self.proj(x)
        for i, c in enumerate(self.convs):
            if i != len(self.convs)-1:
                x = self.leakyRELU(c(x))
            else:
                x = c(x)
        x = torch.mean(x, dim=-1).squeeze(-1)
        return x




class UNet(nn.Module):
    def __init__(self, n_class, fing_size=3):
        super().__init__()
        
        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: 572x572x3

        dims = [1, 64, 128, 256, 512, 1024]
        self.e11 = nn.Conv1d(dims[0], dims[1], kernel_size=3, padding=1) # output: 570x570x64
        self.conv_e11 = nn.Conv1d(fing_size, dims[1]*2, kernel_size=3, padding=1) 
        self.e12 = nn.Conv1d(dims[1], dims[1], kernel_size=3, padding=1) # output: 568x568x64
        self.conv_e12 = nn.Conv1d(fing_size, dims[1]*2, kernel_size=3, padding=1) 
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2) # output: 284x284x64

        # input: 284x284x64
        self.e21 = nn.Conv1d(dims[1], dims[2], kernel_size=3, padding=1) # output: 282x282x128
        self.conv_e21 = nn.Conv1d(fing_size, dims[2]*2, kernel_size=3, padding=1) 
        self.e22 = nn.Conv1d(dims[2], dims[2], kernel_size=3, padding=1) # output: 280x280x128
        self.conv_e22 = nn.Conv1d(fing_size, dims[2]*2, kernel_size=3, padding=1) 
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2) # output: 140x140x128

        # input: 140x140x128
        self.e31 = nn.Conv1d(dims[2], dims[3], kernel_size=3, padding=1) # output: 138x138x256
        self.conv_e31 = nn.Conv1d(fing_size, dims[3]*2, kernel_size=3, padding=1) 
        self.e32 = nn.Conv1d(dims[3], dims[3], kernel_size=3, padding=1) # output: 136x136x256
        self.conv_e32 = nn.Conv1d(fing_size, dims[3]*2, kernel_size=3, padding=1) 
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2) # output: 68x68x256

        # input: 68x68x256
        self.e41 = nn.Conv1d(dims[3], dims[4], kernel_size=3, padding=1) # output: 66x66x512
        self.conv_e41 = nn.Conv1d(fing_size, dims[4]*2, kernel_size=3, padding=1) 
        self.e42 = nn.Conv1d(dims[4], dims[4], kernel_size=3, padding=1) # output: 64x64x512
        self.conv_e42 = nn.Conv1d(fing_size, dims[4]*2, kernel_size=3, padding=1) 
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2) # output: 32x32x512

        # input: 32x32x512
        self.e51 = nn.Conv1d(dims[4], dims[5], kernel_size=3, padding=1) # output: 30x30x1024
        self.conv_e51 = nn.Conv1d(fing_size, dims[5]*2, kernel_size=3, padding=1) 
        self.e52 = nn.Conv1d( dims[5],  dims[5], kernel_size=3, padding=1) # output: 28x28x1024
        self.conv_e52 = nn.Conv1d(fing_size, dims[5]*2, kernel_size=3, padding=1) 


        # Decoder
        self.upconv1 = nn.ConvTranspose1d( dims[5],  dims[4], kernel_size=2, stride=2)
        self.d11 = nn.Conv1d( dims[5],dims[4], kernel_size=3, padding=1)
        self.conv_d11 = nn.Conv1d( fing_size,  dims[4]*2, kernel_size=3, padding=1) # output: 28x28x1024
        self.d12 = nn.Conv1d(dims[4], dims[4], kernel_size=3, padding=1)
        self.conv_d12 = nn.Conv1d(fing_size,  dims[4]*2, kernel_size=3, padding=1) # output: 28x28x1024

        self.upconv2 = nn.ConvTranspose1d(dims[4], dims[3], kernel_size=2, stride=2)
        self.d21 = nn.Conv1d(dims[4], dims[3], kernel_size=3, padding=1)
        self.conv_d21 = nn.Conv1d(fing_size,  dims[3]*2, kernel_size=3, padding=1) # output: 28x28x1024
        self.d22 = nn.Conv1d(dims[3], dims[3], kernel_size=3, padding=1)
        self.conv_d22 = nn.Conv1d(fing_size,  dims[3]*2, kernel_size=3, padding=1) # output: 28x28x1024

        self.upconv3 = nn.ConvTranspose1d(dims[3], dims[2], kernel_size=2, stride=2)
        self.d31 = nn.Conv1d(dims[3], dims[2], kernel_size=3, padding=1)
        self.conv_d31 = nn.Conv1d(fing_size,  dims[2]*2, kernel_size=3, padding=1) # output: 28x28x1024
        self.d32 = nn.Conv1d(dims[2], dims[2], kernel_size=3, padding=1)
        self.conv_d32 = nn.Conv1d(fing_size,  dims[2]*2, kernel_size=3, padding=1) # output: 28x28x1024

        self.upconv4 = nn.ConvTranspose1d(dims[2], dims[1], kernel_size=2, stride=2)
        self.d41 = nn.Conv1d(dims[2], dims[1], kernel_size=3, padding=1)
        self.conv_d41 = nn.Conv1d(fing_size,  dims[1]*2, kernel_size=3, padding=1) # output: 28x28x1024
        self.d42 = nn.Conv1d(dims[1], dims[1], kernel_size=3, padding=1)
        self.conv_d42 = nn.Conv1d(fing_size,  dims[1]*2, kernel_size=3, padding=1) # output: 28x28x1024

        # Output layer
        self.outconv = nn.Conv1d(dims[1], n_class, kernel_size=1)

        self.relu = nn.ReLU()

    def apply_film(self, x, watermark, conv):
        watermark = watermark.unsqueeze(2)
        a,b = torch.chunk(conv(watermark), chunks=2, dim=1)
        # watermark = watermark.expand(-1, -1,x.size(2))
        return x*a+b #TODO: appply sigmoid
    
    def forward(self, x, watermark):
        # Encoder
        xe11 = self.relu(self.e11(x))
        xe11 = self.apply_film(xe11, watermark,self.conv_e11)
        xe12 = self.relu(self.e12(xe11))
        xe12 = self.apply_film(xe12, watermark, self.conv_e12)
        xp1 = self.pool1(xe12)

        xe21 = self.relu(self.e21(xp1))
        xe21 = self.apply_film(xe21,watermark, self.conv_e21)
        xe22 = self.relu(self.e22(xe21))
        xe22 = self.apply_film(xe22, watermark,self.conv_e22)
        xp2 = self.pool2(xe22)

        xe31 = self.relu(self.e31(xp2))
        xe31 = self.apply_film(xe31, watermark,self.conv_e31)
        xe32 = self.relu(self.e32(xe31))
        xe32 = self.apply_film(xe32, watermark,self.conv_e32)
        xp3 = self.pool3(xe32)

        xe41 = self.relu(self.e41(xp3))
        xe41 = self.apply_film(xe41, watermark,self.conv_e41)
        xe42 = self.relu(self.e42(xe41))
        xe42 = self.apply_film(xe42, watermark, self.conv_e42)
        xp4 = self.pool4(xe42)

        xe51 = self.relu(self.e51(xp4))
        xe51 = self.apply_film(xe51, watermark, self.conv_e51)
        xe52 = self.relu(self.e52(xe51))
        xe52 = self.apply_film(xe52, watermark, self.conv_e52)
        
        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = self.relu(self.d11(xu11))
        xd11 = self.apply_film(xd11, watermark, self.conv_d11)
        xd12 = self.relu(self.d12(xd11))
        xd12 = self.apply_film(xd12, watermark, self.conv_d12)

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = self.relu(self.d21(xu22))
        xd21 = self.apply_film(xd21, watermark, self.conv_d21)
        xd22 = self.relu(self.d22(xd21))
        xd22 = self.apply_film(xd22, watermark, self.conv_d22)

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = self.relu(self.d31(xu33))
        xd31 = self.apply_film(xd31, watermark, self.conv_d31)
        xd32 = self.relu(self.d32(xd31))
        xd32 = self.apply_film(xd32, watermark, self.conv_d32)

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = self.relu(self.d41(xu44))
        xd41 = self.apply_film(xd41, watermark, self.conv_d41)
        xd42 = self.relu(self.d42(xd41))
        xd42 = self.apply_film(xd42, watermark, self.conv_d42)

        # Output layer
        out = self.outconv(xd42)

        return out
