from turtle import forward
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import abc

class BaseModel(pl.LightningModule):
    DEBUG_SHAPE = True
    def __init__(self, img_size, learning_rate):
        super().__init__()
        self.img_size = img_size
        self.learning_rate = learning_rate

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_pred = self.forward(x)    
        loss = self.loss(y_pred, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_pred = self.forward(x)    
        loss = self.loss(y_pred, y)
        self.log('valid_loss', loss)
        return loss

    def plot_output_mask(self, data_loader):
        x, y = next(iter(data_loader))
        out = self.forward(x).detach().numpy()[0,0]
        x = x[0,0].detach().numpy()

        fig = plt.figure(figsize=(10,10))
        plt.imshow(x, cmap='gray')
        plt.imshow(out, alpha=0.5, extent=(0, x.shape[0], x.shape[1], 0))

    def plot_activations(self, x):
        activation_number = x.shape[1]
        fig, ax = plt.subplots(1,activation_number, figsize=(5*activation_number,5))
        if activation_number != 1:
            for ia in range(activation_number):
                k = x[0,ia].cpu().detach().numpy()
                ax[ia].imshow(k)
                ax[ia].axis('off')
        else:
            k = x[0,0].cpu().detach().numpy()
            ax.imshow(k)
            ax.axis('off')

        plt.show()

    def test_forward(self):
        h, w = self.img_size
        tensor = torch.rand([1, 1, h, w])
        output = self.forward(tensor)
        print('Output shape:', output.shape)