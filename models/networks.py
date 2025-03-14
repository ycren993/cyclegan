import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import random
import torch.nn.functional as F
from models.Addmodules.ACMix import ACmix
from models.Addmodules.MSDA import MultiDilatelocalAttention
import torchvision.transforms as transforms
from torchsummary import summary
###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = ExpandedUnetGenerator(input_nc, output_nc, 5, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'SE_ResNet_blocks':
        net = Unet_SEA_ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        #修改标签平滑
        target_real_label = random.randint(8, 11) * 0.1
        target_fake_label = random.randint(0, 3) * 0.1
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class Self_Attention(nn.Module):

    def __init__(self, in_dim, activation):
        super(Self_Attention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        ##  下面的query_conv，key_conv，value_conv即对应Wg,Wf,Wh
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)  # 即得到C^ X C
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)  # 即得到C^ X C
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)  # 即得到C X C
        self.gamma = nn.Parameter(torch.zeros(1))  # 这里即是计算最终输出的时候的伽马值，初始化为0

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        ##  下面的proj_query，proj_key都是C^ X C X C X N= C^ X N
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N),permute即为转置
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check，进行点乘操作
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out

class SEA_ResnetBlock_1(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(SEA_ResnetBlock_1, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
        self.self_attention = Self_Attention(dim, 'relu')

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = self.self_attention(x) + self.conv_block(x) + x  # add skip connections
        return out

class Unet_SEA_ResnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(Unet_SEA_ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.pad = nn.ReflectionPad2d(3)
        self.Down_conv1 = nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias)  # 下采样第一层
        self.conv_norm = norm_layer(input_nc)
        self.relu = nn.ReLU(True)
        self.Down_conv2 = nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=use_bias)  # 下采样第二层
        self.SA = Self_Attention(ngf * 2, 'relu')
        self.Down_conv3 = nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=use_bias)  # 下采样第三层
        self.Sa_block_3 = SEA_ResnetBlock_1(ngf * 4, padding_type=padding_type, norm_layer=norm_layer,
                                      use_dropout=use_dropout, use_bias=use_bias)
        self.Sa_resnetblock_1 = SEA_ResnetBlock_1(ngf * 4, padding_type=padding_type, norm_layer=norm_layer,
                                                  use_dropout=use_dropout, use_bias=use_bias)
        self.resnet = ResnetBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)
        self.Up_conv1 = nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1,
                                           bias=use_bias)
        self.Up_conv2 = nn.ConvTranspose2d(ngf * 2 * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1,
                                           bias=use_bias)
        self.Up_conv3 = nn.Conv2d(ngf * 2, output_nc, kernel_size=7, padding=0)
        self.tan = nn.Tanh()

    def forward(self, x):
        x1 = self.relu(self.conv_norm(self.Down_conv1(self.pad(x))))
        x2 = self.relu(self.conv_norm(self.Down_conv2(x1)))
        x3 = self.relu(self.conv_norm(self.Down_conv3(x2)))
        x4 = self.resnet(x3)
        x = torch.cat([x4, x3], 1)
        x = self.relu(self.conv_norm(self.Up_conv1(x)))
        x = torch.cat([x, x2], 1)
        x = self.relu(self.conv_norm(self.Up_conv2(x)))
        x = torch.cat([x, x1], 1)
        x = self.tan(self.Up_conv3(self.pad(x)))
        return x


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)
    '''
        加的下采样CBAM模块
    '''


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class downSamplingConnectionBlock(nn.Module):
    def __init__(self, down_model,inner_nc):
        super(downSamplingConnectionBlock, self).__init__()
        self.attention = CBAM(inner_nc)
        self.down_model = down_model
    def forward(self, x):
        x1 = self.down_model(x)
        x1 = self.attention(x1) + x1
        return x1

class MultiDilationNet(nn.Module):
    def __init__(self, inner_nc, outer_nc,use_bias=False):
        super(MultiDilationNet, self).__init__()
        sub_inner_nc = inner_nc // 4
        sub_outer_nc = outer_nc // 4
        self.sub1 = nn.Conv2d(inner_nc, sub_outer_nc, kernel_size=1, stride=1, padding=0, dilation=1,bias=use_bias)
        self.sub2 = nn.Conv2d(inner_nc, sub_outer_nc, kernel_size=3, stride=1, padding=2, dilation=2,bias=use_bias)
        self.sub3 = nn.Conv2d(inner_nc, sub_outer_nc, kernel_size=3, stride=1, padding=4, dilation=4,bias=use_bias)
        self.sub4 = nn.Conv2d(inner_nc, sub_outer_nc, kernel_size=3, stride=1, padding=8, dilation=8,bias=use_bias)
        self.cbam = CBAM(sub_outer_nc)
        self.conv1x1 = nn.Conv2d(inner_nc, outer_nc,1,bias=use_bias)
        self.batch_norm = nn.BatchNorm2d(inner_nc)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        sub_cbam1 = self.cbam(self.sub1(x))
        sub_cbam2 = self.cbam(self.sub2(x))
        sub_cbam3 = self.cbam(self.sub3(x))
        sub_cbam4 = self.cbam(self.sub4(x))
        x = torch.cat([sub_cbam1, sub_cbam2, sub_cbam3, sub_cbam4],1)
        return x + self.relu(self.batch_norm(self.conv1x1(x)))
import torch.nn.functional as F
class ASPP(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_ch, 256, 1)
        self.conv3x3_1 = nn.Conv2d(in_ch, 256, 3, padding=6, dilation=6)
        self.conv3x3_2 = nn.Conv2d(in_ch, 256, 3, padding=12, dilation=12)
        self.conv3x3_3 = nn.Conv2d(in_ch, 256, 3, padding=18, dilation=18)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.final_conv = nn.Conv2d(4*256+in_ch, in_ch, 1)

    def forward(self, x):
        feat1 = self.conv1x1(x)
        feat2 = self.conv3x3_1(x)
        feat3 = self.conv3x3_2(x)
        feat4 = self.conv3x3_3(x)
        feat5 = F.interpolate(self.pool(x), size=x.shape[-2:], mode='bilinear')
        return self.final_conv(torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1))
class ExpandedUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ExpandedUnetGenerator, self).__init__()

        # 从最内层开始构建
        # 原始结构中各层的参数对应关系：
        # Block1: inner_nc=512, outer_nc=256
        # Block2: inner_nc=256, outer_nc=128
        # Block3: inner_nc=128, outer_nc=64
        # 最外层:  outer_nc=output_nc


        # 最外层
        self.outermost_down = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
        )
        # 中间层3（对应原始结构中的ngf到ngf*2的块）
        self.middle_block_down_3 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(128)

        )
        # 中间层2（对应原始结构中的ngf*2到ngf*4的块）
        self.middle_block_down_2 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(256)

        )
        # 中间层1（对应原始结构中的ngf*4到ngf*8的块）
        self.middle_block_down_1 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(512)
        )

        # 定义最内层块
        self.innermost_down = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
        )
        self.innermost_up = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(512)
        )


        self.middle_block_up_1 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024 , 256, kernel_size=4, stride=2, padding=1, bias=False),  # 512*2=1024
            norm_layer(256)
        )
        self.middle_block_up_2 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1, bias=False),  # 256*2=512
            norm_layer(128)
        )
        self.middle_block_up_3 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 128*2=256
            norm_layer(64)
        )

        self.out_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(64,output_nc, kernel_size=1, stride=1, padding=0),
        )
        self.outermost_up = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(128, output_nc, kernel_size=4, stride=2, padding=1),  # 64*2=128
            nn.Tanh()
        )
        self.convtranspose_11 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512,512,kernel_size=4,stride=2,padding=1,bias=False)
        )

        self.convtranspose_12 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512,512,kernel_size=8,stride=4,padding=2,bias=False)
        )

        self.convtranspose_21 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False)
        )

        self.conv_31 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False)
        )
        self.conv3x3_1 = nn.Conv2d(512 + 512 + 256,512 + 512 + 256,3,1,1,bias=False)
        self.conv1x1_1 = nn.Conv2d(512 + 512 + 256,512,1,1,0,bias=False)
        self.conv3x3_2 = nn.Conv2d(256 + 512 + 512, 256 + 512 + 512,3,1,1,bias=False)
        self.conv1x1_2 = nn.Conv2d(256 + 512 + 512,256,1,1,0,bias=False)

        self.leaky = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(input_nc, 32, kernel_size=9, stride=1, padding=4, bias=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1x3 = nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=False)


    def forward(self, x):

        x1 = self.outermost_down(x)       #64 128 128

        x2 = self.middle_block_down_3(x1) #128 64 64
        x3 = self.middle_block_down_2(x2) #256 32 32
        x4 = self.middle_block_down_1(x3) #512 16 16

        x5 = self.innermost_down(x4)#512 8 8
        x6 = self.innermost_up(x5)  #512 16 16

        concat_1 = self.conv1x1_1(self.relu(self.conv3x3_1(torch.cat([x4,self.convtranspose_11(x5), self.conv_31(x3)],1))))

        x7 = self.middle_block_up_1(torch.cat([x6,concat_1],1))  #(512 + 512 + 512 + 256) * 16 * 16 -> 32 * 32

        concat_2 = self.conv1x1_2(self.relu(self.conv3x3_2(torch.cat([x3,self.convtranspose_11(x4), self.convtranspose_12(x5)], 1))))

        x8 = self.middle_block_up_2(torch.cat([x7,concat_2], 1)) #(256 + 256 + 512 + 512) * 32 * 32 -> 64 * 64
        x9 = self.middle_block_up_3(torch.cat([x8,x2],1)) #(128 + 128) 64 64 -> 128 128
        x10 = self.outermost_up(torch.cat([x9,x1],1))

        return x10,self.out_2(x9)

class ExpandedUnetGenerator1(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ExpandedUnetGenerator1, self).__init__()

        # 从最内层开始构建
        # 原始结构中各层的参数对应关系：
        # Block1: inner_nc=512, outer_nc=256
        # Block2: inner_nc=256, outer_nc=128
        # Block3: inner_nc=128, outer_nc=64
        # 最外层:  outer_nc=output_nc


        # 最外层
        self.outermost_down = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
        )
        # 中间层3（对应原始结构中的ngf到ngf*2的块）
        self.middle_block_down_3 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(128)

        )
        # 中间层2（对应原始结构中的ngf*2到ngf*4的块）
        self.middle_block_down_2 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(256)

        )
        # 中间层1（对应原始结构中的ngf*4到ngf*8的块）
        self.middle_block_down_1 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(512)
        )

        # 定义最内层块
        self.innermost_down = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
        )
        self.innermost_up = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(512)
        )


        self.middle_block_up_1 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024 , 256, kernel_size=4, stride=2, padding=1, bias=False),  # 512*2=1024
            norm_layer(256)
        )
        self.middle_block_up_2 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1, bias=False),  # 256*2=512
            norm_layer(128)
        )
        self.middle_block_up_3 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 128*2=256
            norm_layer(64)
        )

        self.out_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(64,output_nc, kernel_size=1, stride=1, padding=0),
        )
        self.outermost_up = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(128, output_nc, kernel_size=4, stride=2, padding=1),  # 64*2=128
            nn.Tanh()
        )
        self.convtranspose_11 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512,512,kernel_size=4,stride=2,padding=1,bias=False)
        )

        self.convtranspose_12 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512,512,kernel_size=8,stride=4,padding=2,bias=False)
        )

        self.convtranspose_21 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False)
        )

        self.conv_31 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False)
        )
        self.conv3x3_1 = nn.Conv2d(512 + 512 + 256,512 + 512 + 256,3,1,1,bias=False)
        self.conv1x1_1 = nn.Conv2d(512 + 512 + 256,512,1,1,0,bias=False)
        self.conv3x3_2 = nn.Conv2d(256 + 512 + 512, 256 + 512 + 512,3,1,1,bias=False)
        self.conv1x1_2 = nn.Conv2d(256 + 512 + 512,256,1,1,0,bias=False)

        self.leaky = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(input_nc, 32, kernel_size=9, stride=1, padding=4, bias=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1x3 = nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=False)


    def forward(self, x):

        x1 = self.outermost_down(x)       #64 128 128

        x2 = self.middle_block_down_3(x1) #128 64 64
        x3 = self.middle_block_down_2(x2) #256 32 32
        x4 = self.middle_block_down_1(x3) #512 16 16

        x5 = self.innermost_down(x4)#512 8 8
        x6 = self.innermost_up(x5)  #512 16 16

        concat_1 = self.conv1x1_1(self.relu(self.conv3x3_1(torch.cat([x4,F.interpolate(x5,16,mode='bilinear'), F.interpolate(x3,(16,16),mode='bilinear')],1))))

        x7 = self.middle_block_up_1(torch.cat([x6,concat_1],1))  #(512 + 512 + 512 + 256) * 16 * 16 -> 32 * 32

        concat_2 = self.conv1x1_2(self.relu(self.conv3x3_2(torch.cat([x3,F.interpolate(x4,32,mode='bilinear'), F.interpolate(x5,32,mode='bilinear')], 1))))

        x8 = self.middle_block_up_2(torch.cat([x7,concat_2], 1)) #(256 + 256 + 512 + 512) * 32 * 32 -> 64 * 64
        x9 = self.middle_block_up_3(torch.cat([x8,x2],1)) #(128 + 128) 64 64 -> 128 128
        x10 = self.outermost_up(torch.cat([x9,x1],1))

        return x10,self.out_2(x9)

class ExpandedUnetGenerator2(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ExpandedUnetGenerator2, self).__init__()

        # 从最内层开始构建
        # 原始结构中各层的参数对应关系：
        # Block1: inner_nc=512, outer_nc=256
        # Block2: inner_nc=256, outer_nc=128
        # Block3: inner_nc=128, outer_nc=64
        # 最外层:  outer_nc=output_nc


        # 最外层
        self.outermost_down = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
        )
        # 中间层3（对应原始结构中的ngf到ngf*2的块）
        self.middle_block_down_3 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(128)

        )
        # 中间层2（对应原始结构中的ngf*2到ngf*4的块）
        self.middle_block_down_2 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(256)

        )
        # 中间层1（对应原始结构中的ngf*4到ngf*8的块）
        self.middle_block_down_1 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(512)
        )

        # 定义最内层块
        self.innermost_down = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
        )
        self.innermost_up = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(512)
        )


        self.middle_block_up_1 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024 , 256, kernel_size=4, stride=2, padding=1, bias=False),  # 512*2=1024
            norm_layer(256)
        )
        self.middle_block_up_2 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1, bias=False),  # 256*2=512
            norm_layer(128)
        )
        self.middle_block_up_3 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 128*2=256
            norm_layer(64)
        )

        self.out_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(64,output_nc, kernel_size=1, stride=1, padding=0),
        )
        self.outermost_up = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(128, output_nc, kernel_size=4, stride=2, padding=1),  # 64*2=128
            nn.Tanh()
        )
        self.convtranspose_11 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512,512,kernel_size=4,stride=2,padding=1,bias=False)
        )

        self.convtranspose_12 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512,512,kernel_size=8,stride=4,padding=2,bias=False)
        )

        self.convtranspose_21 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False)
        )

        self.conv_31 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False)
        )
        self.conv3x3_1 = nn.Conv2d(512 + 512 + 256,512 + 512 + 256,3,1,1,bias=False)
        self.conv1x1_1 = nn.Conv2d(512 + 512 + 256,512,1,1,0,bias=False)
        self.conv3x3_2 = nn.Conv2d(256 + 512 + 512, 256 + 512 + 512,3,1,1,bias=False)
        self.conv1x1_2 = nn.Conv2d(256 + 512 + 512,256,1,1,0,bias=False)

        self.leaky = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(input_nc, 32, kernel_size=9, stride=1, padding=4, bias=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1x3 = nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=False)


    def forward(self, x):

        x1 = self.outermost_down(x)       #64 128 128

        x2 = self.middle_block_down_3(x1) #128 64 64
        x3 = self.middle_block_down_2(x2) #256 32 32
        x4 = self.middle_block_down_1(x3) #512 16 16

        x5 = self.innermost_down(x4)#512 8 8
        x6 = self.innermost_up(x5)  #512 16 16


        x7 = self.middle_block_up_1(torch.cat([x6,x4],1))  #(512 + 512 + 512 + 256) * 16 * 16 -> 32 * 32

        x8 = self.middle_block_up_2(torch.cat([x7,x3], 1)) #(256 + 256 + 512 + 512) * 32 * 32 -> 64 * 64
        x9 = self.middle_block_up_3(torch.cat([x8,x2],1)) #(128 + 128) 64 64 -> 128 128
        x10 = self.outermost_up(torch.cat([x9,x1],1))

        return x10,self.out_2(x9)
class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        acmix = ACmix(inner_nc, inner_nc)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)


            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            # multidilation = MultiDilationNet(inner_nc, inner_nc, use_bias)
            multinet = MultiDilatelocalAttention(inner_nc)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + [multinet] + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv,acmix, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 256)
    # net = UnetGenerator(3, 3, 7, 64, norm_layer=nn.BatchNorm2d, use_dropout=False)
    # light_net = LightenNet(3)
    #
    # output = net(x)
    # output_light = light_net(x)
    # print('output_light = ', output_light.shape)
    # print('output = ', output.shape)
    # print((output_light * output).shape)
    # net = Unet64Generator(3, 3, 7, 64, norm_layer=nn.BatchNorm2d, use_dropout=False)
    net = ExpandedUnetGenerator1(3, 3, 5, 64, norm_layer=nn.BatchNorm2d, use_dropout=False)
    # output1, output2 = net(x)
    # print(output1.shape, output2.shape)
    print(summary(net.cuda(), input_size=(3, 256, 256)))




    # d_net = NLayerDiscriminator(3, 64, 3, norm_layer=nn.BatchNorm2d)
    # output = d_net(x)
    # print(output.shape)