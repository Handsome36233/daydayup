import types
import numpy as np
from random import shuffle
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import os
import time
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, images_dir, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.files = [os.path.join(images_dir, p) for p in os.listdir(images_dir)]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = cv2.imread(img_path)
        if self.transform:
            img = self.transform(img)
        return img
    

def test_cv2(batch_size=1, num_workers=1, device='cpu'):
    images_dir = "./test_dali/data/dog/"
    dataset = CustomDataset(images_dir=images_dir, transform=None)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # warm up
    for i, batch in enumerate(data_loader):
        if device == 'cpu':
            pass
        else:
            batch.cuda()
    t1 = time.time()
    for i, batch in enumerate(data_loader):
        if device == 'cpu':
            pass
        else:
            batch.cuda()
    t2 = time.time()
    print("Time to load and process batch: ", t2 - t1)



class ExternalInputIterator(object):
    def __init__(self, batch_size):
        self.images_dir = "./test_dali/data/dog/"
        self.batch_size = batch_size
        self.files = [self.images_dir + p for p in os.listdir(self.images_dir)]
        shuffle(self.files)

    def __iter__(self):
        self.i = 0
        self.n = len(self.files)
        return self

    def __next__(self):
        batch = []
        for _ in range(self.batch_size):
            f = open(self.files[self.i], "rb")
            batch.append(np.frombuffer(f.read(), dtype=np.uint8))
            self.i = (self.i + 1) % self.n
        return batch
    

def test_dali(batch_size=1, num_threads=1, device='cpu'):
    eii = ExternalInputIterator(batch_size)
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=0)
    with pipe:
        images = fn.external_source(
            source=eii, dtype=types.UINT8
        )
        images = fn.decoders.image(images, device=device if device == 'cpu' else "mixed", output_type=types.RGB)
        # images = fn.resize(images, device="gpu", resize_x=256, resize_y=256)
        pipe.set_outputs(images)
    pipe.build()
    # warm up
    pipe.run()
    t1 = time.time()
    for _ in range(16//batch_size):
        pipe.run()
    t2 = time.time()
    print("DALI time:", t2 - t1)


test_dali(batch_size=16, num_threads=1, device='gpu')
# test_cv2(batch_size=1, num_workers=1, device='1')
