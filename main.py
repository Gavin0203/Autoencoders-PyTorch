#Autoencoders --> Generative model
#We use the MSE loss function, The computed image pixels should be close to the original image pixel.

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

transform = transforms.ToTensor()

mnist_data = datasets.MNIST(root = './data',train = True, download = True, transform = transform)

data_loader = torch.utils.data.DataLoader(dataset = mnist_data, bathc_size = 64, shuffle = True)


dataiter = iter(data_loader)
images, labels = dataiter.next()
print(torch.min(images),torch.max(images))

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        #The goal is to repeatedly reduce size.
        #Initial size --> N, 785 (N --> Batch size, 28*28 -> Pixels of images in the dataset)
        self.encoder = nn.Sequential(
            nn.Linear(28*28,128), #Reduce size by N,784 --> N,128
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,12),
            nn.ReLU(),
            nn.Linear(12,3), # -> N,3
        )

        self.decoder = nn.Sequential(
            nn.Linear(3,12),
            nn.ReLU(),
            nn.Linear(12,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,28*28), # -> Reduce size by N,3 --> N,784
            nn.Sigmoid() # We use Sigmoid as the value of the image pixel tensors are between 0 and 1.
        )
         

    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
#Note : [-1, 1] -> nn.Tanh  --> Can happen if you use normalisation in the dataloader step as shown above in the data loader 2 section.


model = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-5)

num_epochs = 10 
outputs = []

for epoch in range(num_epochs):
    for (img,_) in data_loader:
        img = img.reshape(-1,28*28)
        recon = model(img)   #reconstructed image
        loss = criterion(recon,img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch : {epoch+1}, Loss : {loss.item():.4f}')
    outputs.appen((epoch, img, recon))