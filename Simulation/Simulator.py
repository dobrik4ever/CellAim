import numpy as np
from skimage import draw

class Simulator:    

    def __init__(self, canvas=(300,300)):
        self.height, self.width = canvas
        self.canvas = np.zeros((self.height, self.width))
        self.mask = np.zeros_like(self.canvas)
        self.ellipse_coordinates = []
        self.layers = []

    def apply_filter(self, filter, args):
        self.canvas = filter(self.canvas, *args)

    def add_noise(self, noise_level):
        self.canvas += np.random.rand(self.height, self.width) * noise_level
    
    def normalize(self):
        self.canvas = self.canvas - self.canvas.min()
        self.canvas = self.canvas / self.canvas.max()

    def add_cells(self, cell_class, n_cells):
        layer = np.zeros_like(self.canvas).astype(bool)
        self.layers.append(layer)
        self.X, self.Y = [], []
        for _ in range(n_cells):
            pos_found = False
            while not pos_found:
                posy = np.random.randint(0, self.height)
                posx = np.random.randint(0, self.width)
                intensity = float(np.random.randint(20,100)) / 100.0
                cell = cell_class(self, (posy, posx), intensity)
                if not np.any(layer[cell.cytoplasm_i] == 1):
                    pos_found = True
                    layer[cell.cytoplasm_i] = cell.intensity
            self.X.append(posx); self.Y.append(posy)
            cell.draw()
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)

class Cell:
    def __init__(self, sim:Simulator, pos:tuple, intensity:float):
        self.sim = sim
        self.y, self.x = pos
        self.intensity = intensity

    def generate_shape(self, borders, shape='ellipse'):
        rmin, rmax = borders
        rx = np.random.randint(rmin, rmax)
        ry = np.random.randint(rmin, rmax)
        rot = np.deg2rad(np.random.randint(0, 360))
        if shape == 'ellipse':
            rr, cc = draw.ellipse(self.y, self.x, ry, rx, rotation = rot, shape=self.sim.canvas.shape)

        return rr, cc

class Ellipse_cell(Cell):
    cytoplasm_r = (5, 20)
    nucleus_r = (2,8)

    def __init__(self, sim:Simulator, pos:tuple, intensity:float):
        super().__init__(sim, pos, intensity)
        self.add_cytoplasm()
        self.add_nucleus()
        self.channels = [
            (self.cytoplasm_i, intensity),
            (self.nucleus_i, 0.0)
        ]

    def draw(self):
        for (channel, value) in self.channels:
            self.sim.canvas[channel] = value

    def add_cytoplasm(self):
        self.cytoplasm_i = self.generate_shape(self.cytoplasm_r)

    def add_nucleus(self):
        self.nucleus_i = self.generate_shape(self.nucleus_r)
