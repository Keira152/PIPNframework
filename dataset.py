from __future__ import print_function
from torch.utils.data import Dataset
import numpy as np
import random
import os

class myData(Dataset):
    """
    Implements a dataset for the temperature field prediction task.

    Args:
        GeometryDir: path to the directory containing the geometric pointclouds.
        SurfaceDir: path to the directory containing the surface temperature field pointclouds.    
    """

    def __init__(self, GeometryDir, SurfaceDir):

        self.root_dir1 = GeometryDir
        self.root_dir2 = SurfaceDir
        self.files1 = []
        self.files2 = []

        for file1 in os.listdir(self.root_dir1):
            o = {}
            o['data'] = self.root_dir1 + file1
            self.files1.append(o)

        for file2 in os.listdir(self.root_dir2):
            s = {}
            s['data'] = self.root_dir2 + file2
            self.files2.append(s)

    def __len__(self) -> int:
        return len(self.files1)
    
    def process(file, is_geometry=True):
        random.seed(0)
        if is_geometry:
            loaddata = np.loadtxt(file, delimiter='\t')
            res = np.asarray(random.choices(loaddata, weights=None, cum_weights=None, k=10000))
            images = res[:, 0:3]
            labels = res[:, 3:4]
            return images, labels
        else:
            loaddata = np.loadtxt(file, delimiter='\t')
            surface = np.asarray(random.choices(loaddata, weights=None, cum_weights=None, k=300))
            return surface

    def __getitem__(self, idx):
        path1 = self.files1[idx]['data']
        path2 = self.files2[idx]['data']
        with open(path1, 'r') as f1:
            xyz, T = self.process(f1, is_geometry=True)
        with open(path2, 'r') as f2:
            surface = self.process(f2, is_geometry=False)

        return {'image': np.array(xyz, dtype="float32"), 'surface': np.array(surface, dtype="float32"), 'label': np.array(T, dtype="float32")}