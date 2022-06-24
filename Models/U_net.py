from Models import BaseModel
import torch
import torchvision
from torch import nn
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pytorch_lightning.callbacks import Callback

class PlottingCallback(Callback):

    def __init__(self, dataloader):
        super().__init__()
        self.dataloader = dataloader

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % 10 == 0:
            image, label = next(iter(self.dataloader))
            output = pl_module.forward(image.cuda())

            image_m = torchvision.utils.make_grid(image, 5)
            output_m = torchvision.utils.make_grid(output, 5)
            image_m = image_m.cpu().detach().numpy()[0]
            output_m = output_m.cpu().detach().numpy()[0]

            fig = plt.figure(figsize=(14,14))
            plt.imshow(image_m, cmap='gray')
            plt.imshow(output_m, cmap='gnuplot2', alpha=0.5)
            plt.axis('off')
            plt.tight_layout()
            fig.savefig(f'Cell_detection.png')
            plt.close()

class Encoder(torch.nn.Module):
    def __init__(self, c_in, c_out, ksize):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(c_in, c_out, ksize, padding=ksize // 2),
            nn.LeakyReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(c_out, c_out, ksize, padding=ksize // 2),
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
            nn.ConvTranspose2d(c_in, c_out, ksize, padding=ksize // 2),
            nn.LeakyReLU()
        )
        self.upconv_2 = nn.Sequential(
            nn.ConvTranspose2d(c_out, c_out, ksize, padding=ksize // 2),
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
        self.enc_1 = Encoder(1, 9, 3)
        self.enc_2 = Encoder(9, 18, 3)
        self.enc_3 = Encoder(18, 36, 3)

        self.dec_3 = Decoder(36, 18, 3)
        self.dec_2 = Decoder(18, 9, 3)
        self.dec_1 = Decoder(9, 1, 3)
        self.smoother = nn.Sequential(
            nn.Conv2d(1,3,3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(3,1,3, padding=1),
            nn.LeakyReLU(),
        )
        self.output_function = nn.LeakyReLU()
        self.save_hyperparameters()
        self.example_input_array = torch.rand([1,1,img_size[0],img_size[1]])

    def loss(self, output, target):
        loss_1 = torch.mean(torch.square(output[target == 0] - target[target == 0]))
        loss_2 = torch.mean(torch.square(output[target == 1] - target[target == 1]))
        loss = loss_1 + loss_2
        return loss 

    def forward(self, x):
        ey1, z1 = self.enc_1(x)
        ey2, z2 = self.enc_2(ey1)
        ey3, z3 = self.enc_3(ey2)
 
        dy3 = self.dec_3(ey3, z3)
        dy2 = self.dec_2(dy3 + ey2, z2)
        dy1 = self.dec_1(dy2 + ey1, z1)
        y = self.output_function(dy1)
        return y

# model = U_net([100,100], learning_rate=1e-1)
# model.forward(torch.rand([1,1,10,10]))