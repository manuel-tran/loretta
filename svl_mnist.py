"""
Refactored code, not fully tested
"""

import numpy as np
import os
import pandas as pd
import random
import torch
import torchvision
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import BertTokenizer

SEED = 0
PATH = '...'


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def align_train(dataset, a_to_b, a_to_c):
    num_cls = 10
    tgt = torch.tensor(dataset.targets)

    for i in range(num_cls):
        idx_cls = (tgt == i).nonzero(as_tuple=True)[0]
        a_to_b += idx_cls.tolist()[0:1200]
        a_to_c += idx_cls.tolist()[1200:2400]

    print(len(a_to_b), len(a_to_c))
    print(len(set(a_to_b + a_to_c)))


def align_test(dataset, a_to_b, a_to_c):
    num_cls = 10
    tgt = torch.tensor(dataset.targets)

    for i in range(num_cls):
        idx_cls = (tgt == i).nonzero(as_tuple=True)[0]
        a_to_b += idx_cls.tolist()[0:1200]
        a_to_c += idx_cls.tolist()[1200:2400]

    print(len(a_to_b), len(a_to_c))
    print(len(set(a_to_b + a_to_c)))


class Tensor2List():
    """
    Converts tensor to list
    """
    def __call__(self, tensor):
        return tensor.tolist()


class Padding:
    """
    Padding a sequence to the maximum length
    """
    def __init__(self, max_len, pad_tok):
        self.max_len = max_len
        self.pad_tok = pad_tok

    def __call__(self, x):
        x += [self.pad_tok] * (self.max_len - len(x))
        return torch.tensor(x)


class ImageTokenizer:
    """
    Image tokenizer
    """
    def __init__(self, height, width, offset=0):
        self.height = height
        self.width = width
        self.offset = offset

    def __call__(self, img):
        img = (255 * img.view(self.height * self.width))
        img = torch.concat(
            [torch.tensor([257]), img,
             torch.tensor([258])], dim=0
        )
        return img.type(torch.LongTensor) + self.offset


class TextTokenizer:
    """
    Text tokenizer
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, txt):
        out = self.tokenizer.encode(
            text=txt,
            padding='do_not_pad',
            max_length=None,
            add_special_tokens=True
        )
        return torch.tensor(out)


class GenericDataset(Dataset):
    """
    Generic dataset
    """
    def __init__(self, dataset, targets, transform):
        self.dataset = dataset
        self.targets = targets
        self.transform = transform

    def __getitem__(self, idx):
        x = self.dataset[idx]
        y = self.targets[idx]
        if self.trf:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.dataset)


class TextDataset(Dataset):
    """
    Text dataset
    """
    def __init__(self, data, targets, offset, transform=None):
        self.data = data
        self.targets = targets
        self.offset = offset
        self.transform = transform

    def __getitem__(self, idx):

        x = self.data[idx]
        y = self.targets[idx]
        if self.transform:
            x = self.transform(x)
        return np.array(x) + self.offset, y

    def __len__(self):
        return len(self.targets)


class AudioDataset(Dataset):
    """
    Audio Dataset Class
    """
    def __init__(self, data, targets, offset, transform=None):
        self.data = data
        self.targets = targets
        self.offset = offset
        self.transform = transform

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        x = x.tolist()
        x = [257] + x + [258]
        x = torch.tensor(x)
        if self.transform:
            x = self.transform(x)
        return x + self.offset, y

    def __len__(self):
        return len(self.data)


class AlignedDataset2M(Dataset):
    """
    Create a dataset of two aligned modalitities
    """
    def __init__(self, dataset_1, dataset_2, transform=None):
        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
        self.transform = transform

        assert (len(dataset_1) == len(dataset_2))

    def __getitem__(self, idx):
        x_1 = self.dataset_1[idx][0]
        x_2 = self.dataset_2[idx][0]
        y_1 = self.dataset_1[idx][1]
        y_2 = self.dataset_2[idx][1]
        assert y_1 == y_2
        return (x_1, x_2), y_1

    def __len__(self):
        return len(self.dataset_1)


class AlignedDataset3M(Dataset):
    """
    Create a dataset of three aligned modalitities
    """
    def __init__(self, dataset_1, dataset_2, missing=0, transform=None):
        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
        self.missing = missing
        self.transform = transform

        assert (len(dataset_1) == len(dataset_2))

    def __getitem__(self, idx):

        x_1 = self.dataset_1[idx][0]
        x_2 = self.dataset_2[idx][0]
        y_1 = self.dataset_1[idx][1]
        y_2 = self.dataset_2[idx][1]
        assert y_1 == y_2
        if self.missing == 0:
            return (float('-inf'), x_1, x_2), y_1
        if self.missing == 1:
            return (x_1, float('-inf'), x_2), y_1
        if self.missing == 2:
            return (x_1, x_2, float('-inf')), y_1

    def __len__(self):
        return len(self.dataset_1)


# -- Image

seed_everything(SEED)

image_dir = './mnist/'
transform = transforms.Compose([transforms.ToTensor()])

image_train = torchvision.datasets.MNIST(
    root=image_dir, train=True, download=True, transform=transform
)
image_test = torchvision.datasets.MNIST(
    root=image_dir, train=False, download=True, transform=transform
)

img_to_txt_train, img_to_wav_train = [], []
align_train(image_train, img_to_txt_train, img_to_wav_train)

np.save(PATH + 'img_to_txt_train.npy', img_to_txt_train)
np.save(PATH + 'img_to_wav_train.npy', img_to_wav_train)

img_to_txt_test, img_to_wav_test = [], []
align_test(image_test, img_to_txt_test, img_to_wav_test)

np.save(PATH + 'img_to_txt_test.npy', img_to_txt_test)
np.save(PATH + 'img_to_wav_test.npy', img_to_wav_test)

img_transform = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.ToTensor(),
        ImageTokenizer(28, 28),
    ]
)

img_dataset = torchvision.datasets.MNIST(
    root=image_dir, train=True, download=True, transform=img_transform
)

# -- Text

#https://www.kaggle.com/datasets/zynicide/wine-reviews

seed_everything(SEED)

wine_map = {
    0: 'Bordeaux-style Red Blend',
    1: 'Cabernet Sauvignon',
    2: 'Chardonnay',
    3: 'Merlot',
    4: 'Pinot Noir',
    5: 'Red Blend',
    6: 'Riesling',
    7: 'Rosé',
    8: 'Sauvignon Blanc',
    9: 'Syrah'
}

df = pd.read_csv('./winemag-data-130k-v2.csv')

x_train, x_test, y_train, y_test = train_test_split(
    df.description, df.variety, test_size=0.2, random_state=0
)

x_train = x_train.values.tolist()
x_test = x_test.values.tolist()

y_train = y_train.values.tolist()
y_test = y_test.values.tolist()

tgt_train, tgt_test = [], []
for wine in list(wine_map.values()):
    idx = np.where(np.array(y_train) == wine)[0]
    tgt_train += list(idx)
for wine in list(wine_map.values()):
    idx = np.where(np.array(y_test) == wine)[0]
    tgt_test += list(idx)

x_train, x_test = np.array(x_train), np.array(x_test)
y_train, y_test = np.array(y_train), np.array(y_test)

x_train, x_test = x_train[tgt_train], x_test[tgt_test]
y_train, y_test = y_train[tgt_train], y_test[tgt_test]

inv_map = dict((str(v), int(k)) for k, v in wine_map.items())
y_train = [inv_map[i] for i in y_train]
y_test = [inv_map[i] for i in y_test]

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
transform = tokenizer.encode

text_train = GenericDataset(x_train, y_train, transform)
text_test = GenericDataset(x_test, y_test, transform)

txt_to_img_train, txt_to_wav_train = [], []
align_train(text_train, txt_to_img_train, txt_to_wav_train)

np.save(PATH + 'txt_to_img_train.npy', txt_to_img_train)
np.save(PATH + 'txt_to_wav_train.npy', txt_to_wav_train)

txt_to_img_test, txt_to_wav_test = [], []
align_test(text_test, txt_to_img_test, txt_to_wav_test)

np.save(PATH + 'txt_to_img_test.npy', txt_to_img_test)
np.save(PATH + 'txt_to_wav_test.npy', txt_to_wav_test)

txt_transform = transforms.Compose(
    [
        TextTokenizer(BertTokenizer.from_pretrained('bert-base-cased')),
        Tensor2List(),
        Padding(212, 0),
    ]
)

text_trainset = TextDataset(x_train, y_train, txt_transform)

# -- Audio

# https://zenodo.org/record/3515935/files/data_sp_train.npy?download=1
# https://zenodo.org/record/3515935/files/data_sp_test.npy?download=1
# https://zenodo.org/record/3515935/files/data_sp_test.npy?download=1
# https://zenodo.org/record/3515935/files/labels_test.npy?download=1

seed_everything(SEED)

x_train = np.load(PATH + 'data_sp_train.npy')
y_train = np.load(PATH + 'labels_train.npy')

x_test = np.load(PATH + 'data_sp_test.npy')
y_test = np.load(PATH + 'labels_test.npy')

x_train, x_index = np.unique(x_train, axis=0, return_index=True)
y_train = y_train[x_index]

x_test, x_index = np.unique(x_test, axis=0, return_index=True)
y_test = y_test[x_index]

x_whole = np.concatenate((x_train, x_test), axis=0)
y_whole = np.concatenate((y_train, y_test), axis=0)

x_train, x_test, y_train, y_test = train_test_split(
    x_whole,
    y_whole,
    train_size=34121,
    test_size=3999,
    random_state=0,
    shuffle=True,
    stratify=y_whole
)

audio_train = GenericDataset(x_train, y_train)
audio_test = GenericDataset(x_test, y_test)

wav_to_img_train, wav_to_txt_train = [], []
align_train(audio_train, wav_to_img_train, wav_to_txt_train)

np.save(PATH + 'wav_to_viz_train.npy', wav_to_img_train)
np.save(PATH + 'wav_to_txt_train.npy', wav_to_txt_train)

wav_to_img_test, wav_to_txt_test = [], []
align_test(audio_test, wav_to_img_test, wav_to_txt_test)

np.save(PATH + 'wav_to_img_test.npy', wav_to_img_test)
np.save(PATH + 'wav_to_txt_test.npy', wav_to_txt_test)

bins = np.linspace(0, 1, 256)
wav_transform = None

x_train = np.digitize(x_train, right=True, bins=bins)
wav_trainset = AudioDataset(x_train, y_train, wav_transform)

# -- Alignment

#img_to_txt_trainset = torchvision.datasets.MNIST(root=img_root, train=True, download=True, transform=img_transform)
#img_to_txt_trainset.data = img_to_txt_trainset.data[img_to_txt_train]
#img_to_txt_trainset.targets = img_to_txt_trainset.targets[img_to_txt_train]

#txt_to_img_trainset = TextDataset(x_txt_train, y_txt_train, txt_transform)
#txt_to_img_trainset.data = txt_to_img_trainset.data[txt_to_img_train]
#txt_to_img_trainset.targets = txt_to_img_trainset.targets[txt_to_img_train]

#img_txt_trainset = AlignedDataset2M(img_to_txt_trainset, txt_to_img_trainset)

# ...
