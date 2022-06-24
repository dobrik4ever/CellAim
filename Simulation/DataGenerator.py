import os
from torch.utils.data import Dataset
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from skimage import filters, draw
import tqdm
from Simulation import Simulator, Ellipse_cell

class DataGenerator(Dataset):
    number_of_classes = 1
    img_size = 200
    def __init__(self, folder, size=None):
        super().__init__()
        self.folder = folder
        if size:
            self.size = size
        else:
            self.size = len(os.listdir(self.folder))//2

    def deal_with_folders(self):
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        else:
            for file in os.listdir(self.folder):
                os.remove(f'{self.folder}/{file}')

    def show_example(self):
        i = 0
        img = np.load(f'{self.folder}/img_{i}.npy')
        coord = np.load(f'{self.folder}/pos_{i}.npy')

        colors = [(0, 0, 0), (0, 1, 0)] # first color is black, last is red
        cm = LinearSegmentedColormap.from_list(
            "Custom", colors, N=100)
        plt.figure(figsize=(5,5))
        plt.imshow(img, cmap='gray')
        plt.imshow(coord, alpha=0.5,cmap=cm, extent = [0,self.img_size, self.img_size, 0])
        plt.show()

    def generate_Unet_points(self, sim, i):
        mask = np.zeros_like(sim.canvas)
        mask[sim.Y, sim.X] = 1
        np.save(f'{self.folder}/img_{i}.npy', sim.canvas)
        np.save(f'{self.folder}/pos_{i}.npy', mask)

    def generate_CellAim(self, sim, i):
        coordinates = coordinates_to_tensor(sim.X, sim.Y, self.output_shape, self.img_size)
        np.save(f'{self.folder}/img_{i}.npy', sim.canvas)
        np.save(f'{self.folder}/pos_{i}.npy', coordinates[2])

    def generate(self):
        self.deal_with_folders()
        cell_N = [1,10]
        for i in tqdm.trange(self.size):
            sim = Simulator(canvas=(self.img_size, self.img_size))
            Ellipse_cell.cytoplasm_r = (10,15)
            Ellipse_cell.nucleus_r = (3,6)

            sim.add_cells(Ellipse_cell, np.random.randint(*cell_N))
            
            sim.apply_filter(filters.gaussian, args=(2,))
            sim.add_noise(0.6)
            sim.normalize()

            self.generate_Unet_points(sim, i)

            

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        canvas = np.load(f'{self.folder}/img_{index}.npy')
        coordinates = np.load(f'{self.folder}/pos_{index}.npy')
        canvas = torch.tensor([canvas]).float()
        coordinates = torch.tensor([coordinates]).float()
        return canvas, coordinates

def coordinates_to_tensor(Ix:np.array, Iy:np.array, S:int, I:int) -> np.array:
    """Utility function to convert coordinates to tensor

    Args:
        Ix (np.array 1d): array of x coordinates
        Iy (np.array 1d): array of y coordinates
        S (int): tensor XY size
        I (int): image XY size

    Returns:
        np.array: 3D tensor of shape (3, S, S), 0 - X, 1 - Y, 2 - objectness
    """

    Ci = Ix * S // I
    Cj = Iy * S // I

    Sx = Ix * S / I - Ci
    Sy = Iy * S / I - Cj

    array = np.zeros([3, S, S])
    array[0,Cj, Ci] = Sx
    array[1,Cj, Ci] = Sy
    array[2,Cj, Ci] = 1

    return array

def tensor_to_coordinates(tensor:np.array, S:int, I:int):
    """Utility function to convert tensor to coordinates

    Args:
        tensor (np.array): 3D tensor of shape (3, S, S), 0 - X, 1 - Y, 2 - objectness
        S (int): tensor XY size
        I (int): image XY size

    Returns:
        (x, y) tuple: 2 arrays of x and y coordinates
    """
    if len(tensor.shape)!=3:
        raise ValueError(f'Tensor shape is not valid, shape={tensor.shape}')
    if tensor.shape != (3, S, S):
        raise ValueError(f'Tensor shape is not valid, shape={tensor.shape}')

    x, y = tensor[0] * I / S, tensor[1] * I / S

    xx, yy = np.meshgrid(np.arange(0, S), np.arange(0, S))
    xx[tensor[2] == 0] = 0
    yy[tensor[2] == 0] = 0
    
    Ix = xx * I // S + x
    Iy = yy * I // S + y

    Ix = Ix[Ix != 0]
    Iy = Iy[Iy != 0]
    return Ix, Iy

if __name__ == '__main__':
    dt = DataGenerator(folder = 'train', size = 10)
    dv = DataGenerator(folder = 'valid', size = 10)
    dt.generate()
    dv.generate()