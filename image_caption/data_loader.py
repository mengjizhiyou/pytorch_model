import os
import pandas as pd
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

"""
将文本转换为数值：
1. 将每个单词映射到一个索引
2. 每个标题的长度应一样， 对较短长度进行填充
"""

"""
python -m spacy download en_core_web_sm # 下载语言模型

or

pip install nltk
pip install spacy==2.3.5
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz
"""

nlp = spacy.load('en_core_web_sm')  # 加载语言模型


class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def word_token(text):
        return [tokenized.text.lower() for tokenized in nlp.tokenizer(text)]

    def build_vocab(self, sentence_list):
        frequency = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.word_token(sentence):
                if word not in frequency:
                    frequency[word] = 1
                else:
                    frequency[word] += 1

                if frequency[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def word2index(self, text):
        tokenized_text = self.word_token(text)
        return [self.stoi[token] if token in self.stoi else self.stoi['<UNK>']
                for token in tokenized_text]

    def index2word(self, index):
        return ' '.join([self.itos[i] for i in index if i not in [0, 1, 2, 3]])


class FDataset(Dataset):
    def __init__(self, img_dir, caption_file, transform=None, freq_threshold=5):
        self.img_dir = img_dir
        self.df = pd.read_csv(caption_file)[:100]
        self.transform = transform

        self.imgs = self.df['image']
        self.captions = self.df['caption']

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.img_dir, img_id)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        word_index = [self.vocab.stoi['<SOS>']] + self.vocab.word2index(caption) + [self.vocab.stoi['<EOS>']]
        return img, torch.tensor(word_index)


class Collate_fn:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        batch.sort(key=lambda item: len(item[1]), reverse=True)
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        lengths = [len(item[1]) for item in batch]
        return imgs, targets, lengths


def get_loader(img_dir, caption_file, transform, batch_size=32, num_workers=0, shuffle=True, pin_memory=True):
    dataset = FDataset(img_dir, caption_file, transform)
    pad_idx = dataset.vocab.stoi['<PAD>']
    loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers,
                        shuffle=shuffle, pin_memory=pin_memory,
                        collate_fn=Collate_fn(pad_idx=pad_idx))
    return dataset, loader


if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset, loader = get_loader('./flickr8k/images', './flickr8k/captions.txt', transform)
    for idx, (imgs, captions, cap_len) in enumerate(loader):
        print(imgs.shape, captions.shape, cap_len)
