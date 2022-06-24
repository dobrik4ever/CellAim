from Models import *
import torch
import os
import unittest  
import logging
import sys

log = logging.getLogger()
log.level = logging.DEBUG
log.addHandler(logging.StreamHandler(sys.stdout))

def model_forward(model):
    h, w = model.img_size
    tensor = torch.rand([2, 1, h, w])
    output = model.forward(tensor)

class Test_models(unittest.TestCase):
    
    def test_unet_forward(self):
        model_forward(model = U_net(img_size=[100,100]))
        
if __name__ == '__main__':
    unittest.main()
    # unittest.TextTestRunner().main()