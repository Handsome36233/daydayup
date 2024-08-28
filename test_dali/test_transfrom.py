import types
import numpy as np
from random import shuffle
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import os
import time
import numpy as np


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
        images = fn.resize(images, device=device, resize_x=2560, resize_y=2560)
        pipe.set_outputs(images)
    pipe.build()
    # warm up
    pipe.run()
    t1 = time.time()
    for _ in range(16//batch_size):
        pipe.run()
    t2 = time.time()
    print("DALI time:", t2 - t1)


test_dali(batch_size=1, num_threads=1, device='cpu')
