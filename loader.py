import random
import numpy as np
from torch.utils.data.sampler import Sampler
import pdb
from collections import defaultdict
import os
import math
import copy
import random
from PIL import Image
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class RandCycleIter:
  '''
  Return data_list per class
  Shuffle the returning order after one epoch
  '''
  def __init__ (self, data, shuffle=True):
    self.data_list = list(data)
    self.length = len(self.data_list)
    self.i = self.length - 1
    self.shuffle = shuffle

  def __iter__ (self):
    return self

  def __next__ (self):
    self.i += 1

    if self.i == self.length:
      self.i = 0
      if self.shuffle:
        random.shuffle(self.data_list)

    return self.data_list[self.i]


class EpisodeSampler(Sampler):
  def __init__(self, way, shot, query, n_epi, ylst):
    self.way = way
    self.query = query
    self.n_inst = shot + query
    self.n_epi = n_epi

    clswise_xidx = defaultdict(list)
    for i, y in enumerate(ylst):
      clswise_xidx[y].append(i)
    self.cws_xidx_iter = [RandCycleIter(cxidx, shuffle=True)
                            for cxidx in clswise_xidx.values()]
    self.n_cls = len(clswise_xidx)

  def __iter__ (self):
    return self.create_episode()

  def __len__ (self):
    return (self.n_epi * self.way * self.n_inst)

  def _sampling_cls(self):
    return torch.randperm(self.n_cls)[:self.way]

  def create_episode(self):
    i, j, e = 0, 0, 0
    nw, ni, ne = self.way, self.n_inst, self.n_epi
    xidx_iter = self.cws_xidx_iter

    while e < ne:
      cls_lst = self._sampling_cls()
      cls_iter = iter(cls_lst)

      i, j = 0, 0
      while i < ni * nw:
        if j >= ni:
          j = 0
        if j == 0:
          didx = next(zip(*[xidx_iter[next(cls_iter)]]* ni))
        yield didx[j]

        i += 1; j+= 1
      e += 1



def get_transforms(mode, imsz):
  # Data transformation with augmentation
  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]
  if mode == 'train':
    T = []
    T.append(transforms.Resize(imsz + 2))
    T.append(transforms.CenterCrop(imsz))
    T.append(transforms.RandomResizedCrop(imsz))
    T.append(transforms.RandomHorizontalFlip())
    T.append(transforms.ColorJitter(
      brightness=0.4, contrast=0.4, saturation=0.4, hue=0))
    T.append(transforms.ToTensor())
    T.append(transforms.Normalize(mean, std))
    return transforms.Compose(T)

  elif mode in ['val', 'test']:
    return transforms.Compose([
      transforms.Resize(imsz),
      transforms.ToTensor(),
      transforms.Normalize(mean, std)
      ])


class Episode:
  def __init__(self):
    self.x = {'s': [], 'q': []}
    self.y = {'s': [], 'q': []}
    self.cls_name = {'s': [], 'q': []}


  def set(self, mode, example):
    x, y, cls_name = example
    self.x[mode].append(x)
    self.y[mode].append(y)
    self.cls_name[mode].append(cls_name)


  def get_episode(self):
    xs, ys = torch.stack(self.x['s']), torch.tensor(self.y['s'])
    cls_name_s = self.cls_name['s']

    xq, yq = torch.stack(self.x['q']), torch.tensor(self.y['q'])
    cls_name_q = self.cls_name['q']
    return xs, ys, xq, yq, cls_name_s, cls_name_q


class Collator:
  def __init__(self, way, shot, query):
    self.way = way
    self.shot = shot
    self.n_inst = shot + query

  def __call__(self, batch):
    nw, ni, ns = self.way, self.n_inst, self.shot
    e = Episode()
    for w in range(nw):
      for example in batch[w*ni:w*ni+ns]:
        e.set('s', example)

      for example in batch[w*ni+ns:(w+1)*ni]:
        e.set('q', example)
    return e.get_episode()


class MetaDataset(Dataset):
  def __init__(self, mode, way, shot, query, n_epi, imsz,
      dpath='/v4/hayeon/data/', dname='miniimgnet'):
    self.dpath = dpath
    self.dname = dname
    self.mode = mode
    self.way = way
    self.shot = shot
    self.query = query
    self.n_inst = self.shot + self.query
    self.n_epi = n_epi
    self.T = get_transforms(mode, imsz)
    data = torch.load(dpath+dname+'/'+mode+'.pt')
    self.x = []
    self.y = []
    for k, v in data.items():
      self.x += v
      self.y += [k] * len(v)
    self.T = None
    self.toPILImage = transforms.ToPILImage()

    self.way_idx = 0
    self.x_idx = 0


  def __len__(self):
    return self.n_epi * (self.way * self.n_inst)


  def __getitem__(self, index):
    x = self.x[index].float()
    if self.T is not None:
      x = self.T(self.toPILImage(x))
    cls_name = self.y[index]
    y = self.way_idx
    self.x_idx += 1
    if self.x_idx == self.n_inst:
      self.way_idx += 1
      self.x_idx = 0
    if self.way_idx == self.way:
      self.way_idx = 0
      self.x_idx = 0
    return x, y, cls_name


def get_loader(mode='train', way=5, 
        shot=5, query=15, n_epi=100, imsz=84,
      dpath='/v4/hayeon/data/', dname='miniimgnet'):
  dataset = MetaDataset(mode, way, shot, query,
                          n_epi, imsz, dpath, dname)
  sampler = EpisodeSampler(
              way, shot, query, n_epi, dataset.y)
  collate_fn = Collator(way, shot, query)
  loader = DataLoader(dataset=dataset,
                      sampler=sampler,
                      batch_size=(shot+query)*way,
                      shuffle=False,
                      collate_fn=collate_fn,
                      num_workers=4)
  return loader

trloader = get_loader()
for i, episode in enumerate(trloader, start=1):
  import pdb; pdb.set_trace()
