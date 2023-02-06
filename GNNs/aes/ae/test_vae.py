import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image

from model import VAE


BATCH_SIZE = 100

# MNIST Dataset
train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)
# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# return reconstruction error + KL divergence losses
def recon_loss(recon_x, x, mu, log_var):
    # binary cross entropy
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # kl_divergence
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


def train(vae, epoch, train_loader, optimizer):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.cuda().view(-1, 784)
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = vae(data)
        loss = recon_loss(recon_batch, data, mu, log_var)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

def test(vae, test_loader):
    vae.eval()
    test_loss= 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.cuda().view(-1, 784)
            recon, mu, log_var = vae(data)
            
            # sum up batch loss
            test_loss += recon_loss(recon, data, mu, log_var).item()
        
    test_loss /= len(test_loader.dataset)
    print('Test set loss: {:.4f}'.format(test_loss))    
    
if __name__ == '__main__':
    vae = VAE(
    in_c=784,
    e_hid_c1=256,
    e_hid_c2=128,
    distr_dim=8,
    d_hid_c1=128,
    d_hid_c2=256
    )
    
    if torch.cuda.is_available():
        vae.cuda()
    optimizer = optim.Adam(vae.parameters())
    
    for epoch in range(1, 31):
        train(vae, epoch, train_loader, optimizer)
        test(vae, test_loader)
        
    with torch.no_grad():
        z = torch.randn(256, 8).cuda()
        sample = vae.decoder(z).cuda()
            
        save_image(sample.view(256, 1, 28, 28), './samples/sample_' + '.png')