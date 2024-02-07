import math
from pathlib import Path
from typing import Union
from random import shuffle

import wavio
import torch
import numpy as np

def np_to_wav(
    x: np.ndarray,
    filename: Union[str, Path],
    rate: int = 48000,
    sampwidth: int = 3,
    scale=None,
    **kwargs
):
    if wavio.__version__ <= "0.0.4" and scale is None:
        scale = "none"
    wavio.write(
        str(filename),
        (np.clip(x, -1.0, 1.0) * (2 ** (8 * sampwidth - 1))).astype(np.int32),
        rate,
        scale=scale,
        sampwidth=sampwidth,
        **kwargs,
    )

def wav_to_tensor(filepath):
    audio_data = wav_to_np(filepath)
    return torch.Tensor(audio_data)

def wav_to_np(filename: Union[str, Path]):
    wav = wavio.read(str(filename))
    bit_depth = wav.sampwidth * 8
    audio_data = wav.data / (2.0 ** (bit_depth - 1))

    return audio_data[:, 0]

class DataGenerator:
    def __init__(self, input_wav, output_wav, input_size=12000, output_size=12000, batch_size=16, device=0, shuffle=True):
        self.input_audio = input_wav
        self.output_audio = output_wav
        assert len(self.input_audio) == len(self.output_audio)

        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_slices = math.floor((len(self.input_audio) - self.input_size) / output_size)
        self.num_batches = math.floor(self.num_slices / batch_size)

        self.device = device
        self.dataset_indexes = np.arange(self.num_slices)

        if(shuffle):
            self.shuffle()

    def __len__(self):
        return math.floor(self.num_slices / self.batch_size)
    
    def __getitem__(self, index):
        x_batch = []
        y_batch = []
        batch_start = index * self.batch_size
        batch_end = batch_start + self.batch_size

        for slice_index in self.dataset_indexes[batch_start:batch_end]:
            x_start = slice_index * self.output_size
            x_end = x_start + self.input_size

            y_start = x_end - self.output_size
            y_end = y_start + self.output_size

            x = self.input_audio[x_start:x_end]
            y = self.output_audio[y_start:y_end]

            x_batch.append(x)
            y_batch.append(y)

        x_batch = torch.stack(x_batch).to(self.device)
        y_batch = torch.stack(y_batch).to(self.device)

        return x_batch, y_batch
    
    def __iter__(self):
        for item in (self[i] for i in range(len(self))):
            yield item
    
    def shuffle(self):
        shuffle(self.dataset_indexes)