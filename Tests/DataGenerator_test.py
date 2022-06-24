from Simulation import *
import os
import unittest  

class Test_folder_creation(unittest.TestCase):
    def test_dir_creation(self):
        dt = DataGenerator('train', 10)
        self.assertTrue(os.path.isdir('train'))

if __name__ == '__main__':
    unittest.main()

