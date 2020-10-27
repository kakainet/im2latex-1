import os
import pickle
from collections import defaultdict
import torch
import glob
import sys
import cv2
import numpy as np
import datetime, pytz
from torchvision import transforms

class Vocab:
    def __init__(self, config):
        '''
            config: pathes, unk_threshold=2
        '''
        self.pathes = config['pathes']
        self.unk_threshold = config.get('unk_threshold', 2)
        self.token2idx = {}
        self.idx2token = {}
        self.load()

    def build(self):
        self.start_token = 0
        self.end_token = 1
        self.pad_token = 2
        self.unk_token = 3
        self.frequency = defaultdict(int)
        self.total = 0
        for path in self.pathes:
            formulas = open(path, 'r')
            lines = formulas.readlines()
            for line in lines:
                tokens = line.rstrip('\n').strip(' ').split()
                for token in tokens:
                    self.frequency[token] += 1
                    self.total += 1
        self.token2idx = {'<f>' : 0, '</f>' : 1, '<pad>' : 2, '<unk>' : 3}
        self.idx2token = {0 : '<f>', 1 : '</f>', 2 : '<pad>', 3 : '<unk>'}
        idx = 4
        for path in self.pathes:
            formulas = open(path, 'r')
            lines = formulas.readlines()
            for line in lines:
                tokens = line.rstrip('\n').strip(' ').split()
                for token in tokens:
                    if self.is_eligible(token) and token not in self.token2idx:
                        self.token2idx[token] = idx
                        self.idx2token[idx] = token
                        idx += 1
        # save vocab
        if not os.path.isdir('vocab'):
            os.mkdir('vocab')
        f = open(os.path.join('vocab', 'vocab.pkl'), 'wb')
        pickle.dump(self, f)
        f.close()

    def is_eligible(self, token):
        if self.frequency[token] >= self.unk_threshold:
            return True
        return False

    def load(self):
        try:
            with open(os.path.join('vocab', 'vocab.pkl'), 'rb') as f:
                vocab = pickle.load(f)
                self.token2idx = vocab.token2idx
                self.idx2token = vocab.idx2token
                self.start_token = vocab.start_token
                self.unk_token = vocab.unk_token
                self.pad_token = vocab.pad_token
                self.end_token = vocab.end_token
                self.frequency = vocab.frequency
                self.total = vocab.total
                print(f'tokeny {self.token2idx}')
        except:
            print('yyyy')
            self.build()

    def formulas2tensor(self, formulas, max_len):
        sz = max_len + 2
        tensor = np.zeros((len(formulas), sz))
        for i in range(tensor.shape[0]):
            tensor[i, 0] = self.start_token
            for j in range(len(formulas[i])):
                tensor[i, j+1] = self.token2idx.get(formulas[i][j], self.unk_token)
            for j in range(len(formulas[i])+1, sz-1):
                tensor[i, j] = self.pad_token
            tensor[i, sz-1] = self.end_token
        return tensor

    def tensor2formula(self, tensor, pretty=False, tags=True):
        if not pretty:
            if tags:
                return ' '.join(self.idx2token[tensor[i]] for i in range(tensor.shape[0]))
            else:
                return ' '.join(self.idx2token[tensor[i]] for i in range(tensor.shape[0])
                                if self.idx2token[tensor[i]] not in ['<f>', '</f>', '<pad>'])
        else:
            s = ' '.join(self.idx2token[tensor[i]] for i in range(tensor.shape[0]))
            end = s.find('</f>')
            if end != -1 : end = end - 1
            s = s[4:end]
            s = s.replace('<pad>', '')
            s = s.replace('<unk>', '')
            return s

class data_loader:
    def __init__(self, vocab, config, train=False):
        '''
            config: vocab, batch_size, images_path, formulas_path=None, sort_by_formulas_len=False, shuffle=False):
        '''
        self.vocab = vocab
        self.batch_size = config['batch_size']
        self.images_path = config['images_path']
        self.formulas_path = config.get('formulas_path', None)
        self.sort_by_formulas_len = config.get('sort_by_formulas_len', False)
        self.shuffle = config.get('shuffle', False)
        self.cursor = 0
        self.images = glob.glob(self.images_path + '/*.png')
        self.images.sort()
        self.images.sort(key=len)
        self.has_label = (self.formulas_path is not None)
        if self.has_label:
            with open(self.formulas_path, 'r') as f:
                self.formulas = [line.rstrip('\n').strip(' ').split()
                                 for line in f.readlines()]
            assert(len(self.images) == len(self.formulas))
            assert(not (self.shuffle and self.sort_by_formulas_len))
            self.max_len = 0
            for formula in self.formulas: self.max_len = max(self.max_len, len(formula))
            if self.sort_by_formulas_len:
                zipped = sorted(enumerate(self.formulas), key=lambda x: len(x[1]), reverse=True)
                idx, self.formulas = zip(*zipped)
                self.images = [self.images[i] for i in idx]
            if self.shuffle:
                idx = np.random.permutation(len(self.formulas))
                self.formulas = [self.formulas[i] for i in idx]
                self.images = [self.images[i] for i in idx]
            if train:
                self.data_transforms = transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    transforms.RandomAffine(
                        degrees=5,
                        translate=(0.1, 0.1),
                        scale=(0.2, 1.1),
                        fillcolor=255
                    ),
                    transforms.ColorJitter(brightness=0.03, saturation=0.03),
                    transforms.ToTensor()
                ])
            else:
                self.data_transforms = transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor()                            
                ])

    def get_next_batch(self):
        current_batch_size = min(self.batch_size, len(self.images)-self.cursor)
        if current_batch_size == 0: end_of_epoch = True
        if self.has_label:
            if self.sort_by_formulas_len:
                max_batch_len = len(self.formulas[self.cursor])
            else:
                max_batch_len = -1
                for i in range(self.cursor, self.cursor+current_batch_size):
                    max_batch_len = max(max_batch_len, len(self.formulas[i]))
            batch_formulas_tensor = self.vocab.formulas2tensor(self.formulas[self.cursor:self.cursor+current_batch_size], max_batch_len)
        batch_imgs = []
        for i in range(current_batch_size):
            img = Image.open(self.images[self.cursor])
            img = self.data_transforms(img)
            batch_imgs.append(img)
            
            end_of_epoch = self.move_cursor()
        if self.has_label:
            return torch.stack(batch_imgs), batch_formulas_tensor, end_of_epoch
        else:
            return torch.stack(batch_imgs), end_of_epoch

    def move_cursor(self):
        self.cursor += 1
        if len(self.images) <= self.cursor:
            self.reset_cursor()
            return True
        return False

    def reset_cursor(self):
        self.cursor = 0

class Logger:
    def __init__(self, config):
        '''
            config: owner, log_dir, timezone
        '''
        self.owner = config['owner']
        self.log_dir = config['log_dir']
        if not os.path.isdir(os.path.dirname(self.log_dir)):
            os.mkdir(os.path.dirname(self.log_dir))
        self.timezone = config['timezone']
        self.log = open(self.log_dir, 'a')

    def __call__(self, message):
        tz = pytz.timezone(self.timezone)
        ts = pytz.utc.localize(datetime.datetime.utcnow()).astimezone(tz).strftime("%Y-%m-%d %H:%M:%S")
        print('[{}], {}: {}'.format(self.owner, ts, message))
        print('[{}], {}: {}'.format(self.owner, ts, message), file=self.log)
