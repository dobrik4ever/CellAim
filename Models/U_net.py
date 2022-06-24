from Models import BaseModel
import torch
from torch import nn
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pytorch_lightning.callbacks import Callback

class PlottingCallback(Callback):

    def __init__(self, dataloader):
        super().__init__()
        self.dataloader = dataloader

    def on_train_epoch_end(self, trainer, pl_module):
        image, label = next(iter(self.dataloader))
        output = pl_module.forward(image.cuda())
        output = output[0,0].cpu().detach().numpy()
        image = image[0,0].cpu().detach().numpy()

        colors = [(0, 0, 0), (0, 1, 0)] # first color is black, last is red
        cm = LinearSegmentedColormap.from_list(
            "Custom", colors, N=100)
        fig = plt.figure(figsize=(10,10))
        plt.imshow(image, cmap='gray')
        # plt.imshow(output, alpha=0.5, cmap=cm)
        plt.imshow(output, alpha=0.5, cmap='gnuplot2')
        fig.savefig('output.png')
        plt.close()

class Encoder(torch.nn.Module):
    def __init__(self, c_in, c_out, ksize):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(c_in, c_out, ksize),
            nn.LeakyReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(c_out, c_out, ksize),
            nn.LeakyReLU(),
         )
        self.pool = nn.MaxPool2d(2, return_indices=True)

    def forward(self, x):
        y = self.conv_1(x); 
        y = self.conv_2(y); 
        y, z = self.pool(y);
        return y, z

class Decoder(torch.nn.Module):

    def __init__(self, c_in, c_out, ksize):
        super().__init__()

        self.unpool = nn.MaxUnpool2d(2)
        self.upconv_1 = nn.Sequential(
            nn.ConvTranspose2d(c_in, c_out, ksize),
            nn.LeakyReLU()
        )
        self.upconv_2 = nn.Sequential(
            nn.ConvTranspose2d(c_out, c_out, ksize),
            nn.LeakyReLU()
        )

    def forward(self, x, ind):
        y = self.unpool(x, ind);
        y = self.upconv_1(y);   
        y = self.upconv_2(y);   
        return y

class U_net(BaseModel):

    def __init__(self, img_size, learning_rate):
        super().__init__(img_size, learning_rate)
        self.enc_1 = Encoder(1, 9, 9)
        self.enc_2 = Encoder(9, 18, 5)
        self.dec_2 = Decoder(18, 9, 5)
        self.dec_1 = Decoder(9, 1, 9)
        self.smoother = nn.Sequential(
            nn.Conv2d(1,3,3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(3,1,3, padding=1),
            nn.LeakyReLU(),
        )
        self.output_function = nn.LeakyReLU()
        self.batch_size = 10
        self.save_hyperparameters()
        self.example_input_array = torch.rand([1,1,200,200])

    def loss(self, output, target):
        loss_1 = torch.mean(torch.square(output[target == 0] - target[target == 0]))
        loss_2 = torch.mean(torch.square(output[target == 1] - target[target == 1]))
        loss = loss_1 + loss_2
        return loss 

    def forward(self, x):
        y1, z1 = self.enc_1(x)
        y2, z2 = self.enc_2(y1)
        y3 = self.dec_2(y2, z2)
        y4 = self.dec_1(y3+y1, z1)
        y = self.output_function(y4)
        return y

# model = U_net([100,100], learning_rate=1e-1)
# model.forward(torch.rand([1,1,10,10]))