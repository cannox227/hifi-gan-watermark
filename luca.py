import torch
import torch.nn as nn
import torch.optim as optim

def generate_random_fingerprints(batch_size, num_bits):
    # Genera fingerprint casuali con la forma (batch_size, 1, 2)
    fingerprint = torch.bernoulli(0.5 * torch.ones((batch_size, num_bits, 1)))
    return fingerprint

class MLPenc(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLPenc, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1, num_bits, time)

#input_tensor = torch.randn(batch_size, num_bits, 1)
#input_tensor_reshaped = input_tensor.view(batch_size, num_bits)
#output = model(input_tensor_reshaped)

#print("Shape dell'input:", input_tensor.shape)
#print("Shape dell'output:", output.
class MLPdec(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLPdec, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Appiattisce il tensore (batch_size, (num_bits+1)*time)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Creazione del modello MLP
#model = MLPdec(input_size=(num_bits + 1) * time, output_size=num_bits)

# Generazione di un tensore di esempio con shape (batch_size, num_bits+1, time)
#input_tensor = torch.randn(batch_size, num_bits + 1, time)

# Applicazione del modello sull'input
#output = model(input_tensor)

# Verifica delle forme
#print("Shape dell'input:", input_tensor.shape)
#print("Shape dell'output:", output.shape)

class ConvNetDop(nn.Module):
    def __init__(self):
        super(ConvNetDop, self).__init__()
        # Convolutional layers with "same" padding
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)  # Layer 1
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)  # Layer 2
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)  # Layer 3
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)  # Layer 4
        self.conv5 = nn.Conv1d(128, 256, kernel_size=3, padding=1)  # Layer 5
        self.conv6 = nn.Conv1d(256, 256, kernel_size=3, padding=1)  # Layer 6
        self.conv7 = nn.Conv1d(256, 512, kernel_size=3, padding=1)  # Layer 7
        self.conv8 = nn.Conv1d(512, 512, kernel_size=3, padding=1)  # Layer 8
        self.conv9 = nn.Conv1d(512, 1024, kernel_size=3, padding=1)  # Layer 9
        self.conv10 = nn.Conv1d(1024, 1024, kernel_size=3, padding=1)  # Layer 10

        # Average pooling layer
        self.avg_pool = nn.AvgPool1d(kernel_size=8000)

        # 1x1 convolution to change the number of channels
        self.conv1x1 = nn.Conv1d(1024, 2, kernel_size=1)  # Adjust channels as needed

    def forward(self, x):
        # Input x has shape (batch, channels, time)

        # Apply convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)

        # Apply average pooling
        x = self.avg_pool(x)

        # Apply 1x1 convolution
        x = self.conv1x1(x)

        # Reshape to (batch, channels, time)
        #x = x.view(x.size(0), 256, 1)  # Adjust channels as needed

        return x

class ConvNetDop(nn.Module):
    def __init__(self, num_bits):
        super(ConvNetDop, self).__init__()
        # Convolutional layers with "same" padding
        self.conv1 = nn.Conv1d(num_bits + 1, 64, kernel_size=3, padding=1)  # Layer 1
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)  # Layer 2
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)  # Layer 3
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)  # Layer 4
        self.conv5 = nn.Conv1d(512, 1024, kernel_size=3, padding=1)  # Layer 5

        # Average pooling layer
        self.avg_pool = nn.AvgPool1d(kernel_size=8000)

        # 1x1 convolution to change the number of channels
        self.conv1x1 = nn.Conv1d(1024, num_bits, kernel_size=1)  # Adjust channels for the desired number of output bits

    def forward(self, x):
        # Apply convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # Apply average pooling
        x = self.avg_pool(x)

        # Apply 1x1 convolution
        x = self.conv1x1(x)

        # Reshape to (batch, num_bits)
        x = x.squeeze(dim=2)

        return x

# Check if a CUDA-enabled GPU is available
if torch.cuda.is_available():
    # Use the first available GPU (device 0)
    device = torch.device('cuda:0')
    print("CUDA is available. Running on GPU.")
else:
    # If no GPU is available, use the CPU
    device = torch.device('cpu')
    print("CUDA is not available. Running on CPU.")


batch_size = 16
num_bits = 8
time = 8000
num_epochs = 100000
#model = MLP(num_bits, num_bits * time)
encoder = MLPenc(num_bits, num_bits * time).to(device)
decoder = MLPdec(input_size=(num_bits + 1) * time, output_size=num_bits).to(device)
decoder_conv = ConvNetDop(num_bits).to(device)

criterion = nn.BCELoss().to(device)

optimizer_encoder = optim.AdamW(encoder.parameters(), lr=0.001)
optimizer_decoder = optim.AdamW(encoder.parameters(), lr=0.001)
optimizer_dec_conv = optim.AdamW(encoder.parameters(), lr=0.001)


for epoch in range(num_epochs):
    fingerprint = generate_random_fingerprints(batch_size, num_bits).to(device)
    #print("fingerprint original",fingerprint.shape)
    fingerprint_reshaped = fingerprint.view(batch_size, num_bits)
    #print("fingerprint reshaped",fingerprint_reshaped.shape)
    fingerprint_exp = encoder(fingerprint_reshaped)
    #print("fingerprint exp",fingerprint_exp.shape)

    input_data = torch.randn(batch_size, 1, time).to(device)
    #print("input_data", input_data.shape)

    concatenation = torch.cat((input_data, fingerprint_exp), dim=1)
    #print("concat shape", concatenation.shape)
    #output = decoder(concatenation)
    output = decoder_conv(concatenation)

    output_sigmoid = torch.sigmoid(output)
    #print("output sig",output_sigmoid.shape)

    loss = criterion(output_sigmoid, fingerprint.squeeze(-1))

    #print("fingerprint",fingerprint.squeeze(-1))
    #print("--------------")
    #print("output", output_sigmoid)

    optimizer_encoder.zero_grad()
    #optimizer_decoder.zero_grad()
    optimizer_dec_conv.zero_grad()
    loss.backward()
    optimizer_encoder.step()
    #optimizer_decoder.step()
    optimizer_dec_conv.step()


    if(epoch + 1) % 10 == 0:
      print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')