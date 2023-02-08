# from torch.utils.data import Dataset
# import torch
# from pathlib import Path
# from skimage.io import imread
# from skimage.color import gray2rgb
# import numpy as np
# import pandas as pd
# import torchvision as tv
# import torchvision.transforms as tvtf
#
# train_mean = [0.59685254, 0.59685254, 0.59685254]
# train_std = [0.16043035, 0.16043035, 0.16043035]
#
#
# class ChallengeDataset(Dataset):
#     # TODO implement the Dataset class according to the description
#     def __init__(self, data: pd.DataFrame, mode: str):
#         self.data = data
#         self.mode = mode
#
#         self.images = []
#         for img_path in self.data['filename'].tolist():
#             img_path = Path(img_path)
#             img = imread(img_path)
#             self.images.append(img)
#
#         self._transform = tvtf.Compose([
#             tvtf.ToPILImage(),
#             tvtf.ToTensor(),
#             tvtf.Normalize(train_mean, train_std)
#         ])
#
#     def __len__(self):
#         # return self.data.shape[0]
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         # img_path = Path(self.data.iloc[idx, 0])
#         # img = imread(img_path)
#         # if len(img.shape) == 2:
#         img = self.images[idx]
#         img = gray2rgb(img)
#         img = self._transform(img)
#         # label = self.data.iloc[idx, 1:].to_numpy()
#         # label = np.array(self.data.iloc[idx, 1:]).reshape(1,-1)
#         # label = np.array(self.data[['crack','inactive']].iloc(idx))
#         # label = self.labels[idx,:].astype('float')
#         label = np.array(self.data.iloc[idx, 1:]).astype('float')
#         # print(type(img), type(label))
#         return img, torch.Tensor(label)
import pandas as pd
from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
import os

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(self, data, mode):
        self.data = data
        self.mode = mode
        self._transform = {'train': tv.transforms.Compose([

            tv.transforms.ToPILImage(),
            # tv.transforms.Resize((227,227)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(train_mean, train_std)]),
            'val': tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                # tv.transforms.Resize((227,227)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(train_mean, train_std)])}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = self.data.iloc[index, 0]
        label = np.array(self.data.iloc[index, 1:]).astype('float')

        image = imread(image_path)
        image = gray2rgb(image)
        image = self._transform[self.mode](image)
        return torch.Tensor(image), torch.Tensor(label)



