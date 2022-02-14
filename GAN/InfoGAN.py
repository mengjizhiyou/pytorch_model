"""
Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, Pieter Abbeel

InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets

https://arxiv.org/abs/1606.03657
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
import torch.optim as optim
import torchvision


class Generator(nn.Module):
    def __init__(self, channel_label, channel_continue, channel_noise, channels_img):
        super().__init__()
        channel_input = channel_label + channel_continue + channel_noise
        self.feature = nn.Sequential(
            self.block(channel_input, 1024, 1, 1),
            self.block(1024, 128, 7, 1),
            self.block(128, 64, 4, 2, 1))
        self.last = nn.ConvTranspose2d(64, channels_img, 4, 2, padding=1, bias=False)

    def block(self, in_channel, out_channel, kernel_size, stride=1, padding=0):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return torch.sigmoid(self.last(self.feature(x)))


class Discriminator(nn.Module):
    def __init__(self, channels_img):
        super().__init__()
        self.conv1 = nn.Conv2d(channels_img, 64, 4, 2, 1)
        self.conv2 = self.block(64, 128, 4, 2, 1)
        self.conv3 = self.block(128, 1024, 7)

    def block(self, in_channel, out_channel, kernel_size, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(D, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 1)

    def forward(self, x):
        return torch.sigmoid(self.conv(x))


class Q(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Q, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True))

        self.conv_disc = nn.Conv2d(128, out_channel, 1)
        self.conv_mu = nn.Conv2d(128, 2, 1)
        self.conv_var = nn.Conv2d(128, 2, 1)

    def forward(self, x):
        x = self.conv1(x)
        disc_logits = self.conv_disc(x).squeeze()
        mu = self.conv_mu(x).squeeze()
        var = torch.exp(self.conv_var(x).squeeze())
        return disc_logits, mu, var


def test():
    N, in_channels, H, W = 8, 1, 28, 28
    channel_label, channel_continue, channel_noise = 10, 2, 62
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels)
    assert disc(x).shape == (N, 1024, 1, 1), "Discriminator test failed"
    d = D(1024, 1)
    assert d(disc(x)).shape == (N, 1, 1, 1), "D test failed"
    q = Q(1024, 10)
    assert q(disc(x))[0].shape == (N, 10), "Q test failed"
    gen = Generator(channel_label, channel_continue, channel_noise, in_channels)
    z = torch.randn((N, channel_label + channel_continue + channel_noise, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"


# ----------------------------------------------------------------
class LogGaussianLoss:
    """
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised.
    Treating Q(cj | x) as a factored Gaussian.
    """

    def __call__(self, x, mu, var):
        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
        return logli.sum(1).mean().mul(-1)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# ----------------------------------------------------------------
def noise_sample(batch_size, channel_label, channel_continue, channel_noise):
    #  62, 10, 2
    z = torch.randn(batch_size, channel_noise, 1, 1, device=device)
    dis_c = torch.zeros((batch_size, channel_label,1,1), device=device)
    idx = np.random.randint(channel_label, size=batch_size)
    dis_c[torch.arange(batch_size), idx] = 1.0

    # Random uniform between -1 and 1.
    con_c = torch.rand(batch_size, channel_continue, 1, 1, device=device) * 2 - 1
    noise = torch.cat((z, dis_c), dim=1)
    noise = torch.cat((noise, con_c), dim=1)
    return noise, idx


def to_categorical(dim_label):
    """Returns one-hot encoded Variable"""
    idx = np.array([num for _ in range(dim_label) for num in range(dim_label)])
    category = np.zeros((idx.shape[0], dim_label))
    category[range(idx.shape[0]), idx] = 1.0
    return torch.from_numpy(category)


def fixed_noise(dim_label, dim_continue, dim_noise):
    z = torch.randn(100, dim_noise, 1, 1, device=device)  # noise
    fixed_noise = z
    dis_c = to_categorical(dim_label)  # label: discrete variale
    dis_c = dis_c.contiguous().view(100, -1, 1, 1)
    fixed_noise = torch.cat((fixed_noise, dis_c), dim=1)
    con_c = torch.rand(100, dim_continue, 1, 1, device=device) * 2 - 1  # mu, sigma: continue variable
    fixed_noise = torch.cat((fixed_noise, con_c), dim=1)
    return fixed_noise.float()


def train(dataloader, channel_label, channel_continue, channel_noise, channels_img, lr):
    criterionD = nn.BCELoss()  # Loss for discrimination between real and fake images.
    criterionQ_dis = nn.CrossEntropyLoss()  # Loss for discrete latent code.
    criterionQ_con = LogGaussianLoss()  # Loss for continuous latent code.

    gen = Generator(channel_label, channel_continue, channel_noise, channels_img)
    disc = Discriminator(channels_img)
    d = D(1024, 1)
    q = Q(1024, 10)

    disc.apply(weights_init_normal)
    gen.apply(weights_init_normal)
    d.apply(weights_init_normal)
    q.apply(weights_init_normal)

    opt_gen = optim.Adam([{'params': gen.parameters()}, {'params': q.parameters()}],
                         lr=lr, betas=(0.5, 0.999))
    opt_desc = optim.Adam([{'params': disc.parameters()}, {'params': d.parameters()}],
                          lr=lr, betas=(0.5, 0.999))
    # opt_infp = torch.optim.Adam(
    # itertools.chain(generator.parameters(), discriminator.parameters()), lr=lr, betas=(0.5, 0.999))

    writer_fake = SummaryWriter(f"logs/fake")
    writer_real = SummaryWriter(f"logs/real")
    step = 0
    for epoch in range(10):
        # Target labels not needed! <3 unsupervised
        for batch_idx, (real, _) in enumerate(dataloader):
            batch_size = real.size(0)
            real = real.to(device)

            ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            disc_real = disc(real)
            prob_real = d(disc_real).reshape(-1)
            loss_real = criterionD(prob_real, torch.ones_like(prob_real))

            noise, idx = noise_sample(batch_size, dim_label, dim_continue, dim_noise)
            fake = gen(noise.to(device))
            disc_fake = disc(fake.detach())
            probs_fake = d(disc_fake).view(-1)
            loss_fake = criterionD(probs_fake, torch.zeros_like(probs_fake))

            loss_disc = (loss_real + loss_fake) / 2

            opt_desc.zero_grad()
            loss_disc.backward(retain_graph=True)
            opt_desc.step()

            ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))

            output = disc(fake)
            out_d = d(output).view(-1)
            loss_gen = criterionD(out_d, torch.ones_like(out_d))

            q_logits, q_mu, q_var = q(output)
            target = torch.LongTensor(idx).to(device)
            # Calculating loss for discrete latent code.
            dis_loss = criterionQ_dis(q_logits, target)
            con_loss = criterionQ_con(noise[:, dim_noise + dim_label:].view(-1,2), q_mu, q_var) * 0.1
            # Net loss for generator.
            G_loss = loss_gen + dis_loss + con_loss

            opt_gen.zero_grad()
            # Calculate gradients.
            G_loss.backward()
            # Update parameters.
            opt_gen.step()

            # Print losses occasionally and print to tensorboard
            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch}/{10}] Batch {batch_idx}/{len(dataloader)} \
                      Loss D: {loss_disc:.4f}, loss G: {G_loss:.4f}"
                )

                with torch.no_grad():
                    fake = gen(fixed_noise(dim_label, dim_continue, dim_noise))
                    # take out (up to) 32 examples
                    img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                    img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                step += 1


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
if __name__ == '__main__':
    # dim_noise : dimension of incompressible noise.
    # num_discrete_var : number of discrete latent code used.
    # dim_label : dimension of discrete latent code.
    # dim_continue : number of continuous latent code used.

    N, in_channels, H, W = 8, 1, 28, 28
    dim_label, dim_continue, dim_noise = 10, 2, 62
    transforms = transforms.Compose(
        [transforms.Resize(28),
         transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

    dataset = datasets.MNIST(root="/", transform=transforms, download=True)
    loader = DataLoader(dataset, batch_size=N, shuffle=True)
    train(loader, dim_label, dim_continue, dim_noise, in_channels, lr=0.0002)
    # test()
