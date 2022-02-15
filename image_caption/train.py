from model import *
from torch.nn.utils.rnn import pack_padded_sequence
from utils import *
from nltk.translate.bleu_score import corpus_bleu
from PIL import Image


def train(train_loader, args):
    net = CNNtoRNN(args.attn_dim, args.embed_dim, args.decoder_dim, args.vocab_size, args.encoder_dim, args.dropout)
    net.train()
    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, net.encoder.parameters()), lr=1e-4)
    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, net.decoder.parameters()), lr=4e-4)
    criterion = nn.CrossEntropyLoss().to(device)
    for epoch in range(args.epochs):
        # 调整优化器的学习率
        adjust_learning_rate(encoder_optimizer, 0.8)
        adjust_learning_rate(decoder_optimizer, 0.8)

        for i, (imgs, caps, caplens) in enumerate(train_loader):
            predictions, encoded_captions, decode_len, alphas = net(imgs, caps, caplens)
            predictions = pack_padded_sequence(predictions, decode_len, batch_first=True).data


            targets = encoded_captions[:, 1:]
            targets = pack_padded_sequence(targets, decode_len, batch_first=True).data

            loss = criterion(predictions, targets)
            loss += args.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            decoder_optimizer.zero_grad()
            encoder_optimizer.zero_grad()
            loss.backward()

            if args.grad_clip is not None:
                clip_gradient(decoder_optimizer, args.grad_clip)
                clip_gradient(encoder_optimizer, args.grad_clip)

            decoder_optimizer.step()
            encoder_optimizer.step()
    if args.save_model:
        checkpoint = {
            "state_dict": net.state_dict(),
            "decoder_optimizer": decoder_optimizer.state_dict(),
            "encoder_optimizer": encoder_optimizer.state_dict()}
        save(checkpoint)


def evaluate(net, transform, args):
    net.eval()
    test_img1 = transform(Image.open("test_examples/dog.jpg").convert("RGB")).unsqueeze(0)
    print("Example 1 CORRECT: Dog on a beach by the ocean")
    prediction, alpha = net.caption_image(test_img1, args.index2word)
    print("Example 1 OUTPUT: " + " ".join(prediction))
    test_img1 = transform(Image.open("test_examples/child.jpg").convert("RGB")).unsqueeze(0)
    print("Example 2 CORRECT: Child holding red frisbee outdoors")
    prediction, alpha = net.caption_image(test_img1, args.index2word)
    print("Example 2 OUTPUT: " + " ".join(prediction))
