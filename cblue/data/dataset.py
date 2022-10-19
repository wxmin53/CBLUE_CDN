import numpy as np
import torch
from torch.utils.data import Dataset
from cblue.models import convert_examples_to_features_for_tokens


class CDNDataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            mode='train',
            dtype='cls'
    ):
        super(CDNDataset, self).__init__()

        self.text1 = samples['text1']

        if dtype == 'cls':
            self.text2 = samples['text2']
            if mode != 'test':
                self.label = samples['label']
        else:
            if mode != 'test':
                self.label = samples['label']

        self.data_processor = data_processor
        self.dtype = dtype
        self.mode = mode

    def __getitem__(self, item):
        if self.dtype == 'cls':
            if self.mode != 'test':
                return self.text1[item], self.text2[item], self.label[item]
            else:
                return self.text1[item], self.text2[item]
        else:
            if self.mode != 'test':
                return self.text1[item], self.label[item]
            else:
                return self.text1[item]

    def __len__(self):
        return len(self.text1)
