"""
Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard Zemel, Yoshua Bengio
https://arxiv.org/abs/1502.03044
"""

import torch
import torch.nn as nn
import torchvision.models as models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderCNN(nn.Module):
    def __init__(self, embed_size=14):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d((embed_size, embed_size))
        self.fine_tune()

    def fine_tune(self, fine_tune=True):
        for p in self.resnet.parameters():
            p.requires_grad = True
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

    def forward(self, images):  # 32, 3, 224, 224
        out = self.resnet(images)  # 32, 2048, 7, 7
        out = self.pool(out)  # 32, 2048, 14, 14
        return out.permute(0, 2, 3, 1)  # 32, 14, 14, 2048


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attn_dim):
        super(Attention, self).__init__()
        self.encoder_attn = nn.Linear(encoder_dim, attn_dim)
        self.decoder_attn = nn.Linear(decoder_dim, attn_dim)
        self.linear = nn.Linear(attn_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        attn1 = self.encoder_attn(encoder_out)  # # 32,196,2048 -> 32, 196, 128
        attn2 = self.decoder_attn(decoder_hidden)  # 32, 128
        attn = self.linear(self.relu(attn1 + attn2.unsqueeze(1))).squeeze(2)  # 32, 196
        alpha = self.softmax(attn)  # 32, 196
        attn = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # 32, 2048
        return attn, alpha


class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        super(AttnDecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.attention = Attention(encoder_dim, decoder_dim, attn_dim)
        self.embedding = nn.Embedding(vocab_size, encoder_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTMCell(encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)  # 32, 14*14, 3
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_len):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)
        embedding = self.embedding(encoded_captions)  # 32, 23, 2048
        h, c = self.init_hidden_state(encoder_out)  # 32, 128 // 32, 128
        decode_len = [c - 1 for c in caption_len]
        predictions = torch.zeros(batch_size, max(decode_len), self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_len), num_pixels).to(device)

        for t in range(max(decode_len)):
            batch_size_t = sum([l > t for l in decode_len])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])  # 32,2048
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # 32,2048
            attention_weighted_encoding = gate * attention_weighted_encoding  # 32,2048
            h, c = self.lstm(embedding[:batch_size_t, t, :] + attention_weighted_encoding,
                             (h[:batch_size_t], c[:batch_size_t]))
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_len, alphas


class CNNtoRNN(nn.Module):
    def __init__(self, attn_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        super(CNNtoRNN, self).__init__()
        self.encoder = EncoderCNN().to(device)
        self.decoder = AttnDecoderRNN(attn_dim, embed_dim, decoder_dim, vocab_size, encoder_dim, dropout).to(device)

    def forward(self, images, encoded_captions, caption_len):
        encoder_out = self.encoder(images)
        predictions, encoded_captions, decode_len, alphas = self.decoder(encoder_out, encoded_captions, caption_len)
        return predictions, encoded_captions, decode_len, alphas

    def caption_image(self, image, index2word, max_len=50):
        predictions = []
        alphas = []
        with torch.no_grad():
            encoder_out = self.encoder(image)
            batch_size = encoder_out.size(0)
            encoder_dim = encoder_out.size(-1)
            encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
            h, c = self.decoder.init_hidden_state(encoder_out)
            for _ in range(max_len):
                attention_weighted_encoding, alpha = self.decoder.attention(encoder_out, h)
                gate = self.decoder.sigmoid(self.decoder.f_beta(h))
                attention_weighted_encoding = gate * attention_weighted_encoding  # [1, 2048]
                h, c = self.decoder.lstm(attention_weighted_encoding, (h, c))
                pred = self.decoder.fc(self.decoder.dropout(h)) # 1, 50
                pred = [pred.argmax(1).item()]
                predictions.append(index2word(pred))
                alphas.append(alpha.cpu().detach().numpy())
                if index2word(pred)=="<EOS>":
                    break

        return predictions, alphas
