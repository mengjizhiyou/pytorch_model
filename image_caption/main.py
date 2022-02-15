from data_loader import *
from model import *
from utils import *
from train import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--embed_dim', type=int, default=128)
parser.add_argument('--attn_dim', type=int, default=128)
parser.add_argument('--decoder_dim', type=int, default=128)
parser.add_argument('--encoder_dim', type=int, default=2048)
parser.add_argument('--dropout', type=float , default=0.5)
parser.add_argument('--alpha_c', type=float, default=1.)
parser.add_argument('--grad_clip', type=float, default=5.)
parser.add_argument('--save_model', type=bool, default=True)
parser.add_argument('--load_model', type=bool, default=True)
args = parser.parse_args()

transform = transforms.Compose(
    [
        transforms.Resize((356, 356)),
        transforms.RandomCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


dataset, loader = get_loader('./flickr8k/images', './flickr8k/captions.txt', transform)

args.vocab_size = len(dataset.vocab)
args.index2word = dataset.vocab.index2word

# train(loader, args)

if args.load_model:
    net = CNNtoRNN(args.attn_dim, args.embed_dim, args.decoder_dim, args.vocab_size, args.encoder_dim, args.dropout)
    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, net.encoder.parameters()),lr=1e-4)
    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, net.decoder.parameters()),lr=4e-4)
    load("./my_checkpoint.pth.rar", net, decoder_optimizer, encoder_optimizer)
    evaluate(net, transform, args)
