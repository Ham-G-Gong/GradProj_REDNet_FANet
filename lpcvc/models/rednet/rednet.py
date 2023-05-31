import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import Resnet18
from lpcvc.loss import OhemCELoss2D,CrossEntropyLoss

up_kwargs = {'mode': 'bilinear', 'align_corners': True}


class BatchNorm2d(nn.BatchNorm2d):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, num_features, activation='none'):
        super(BatchNorm2d, self).__init__(num_features=num_features)
        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'none':
            self.activation = lambda x:x
        else:
            raise Exception("Accepted activation: ['leaky_relu']")

    def forward(self, x):
        return self.activation(super(BatchNorm2d, self).forward(x))

class REDNet(nn.Module):
    def __init__(self,
                 nclass=14,
                 backbone='resnet18',
                 norm_layer=BatchNorm2d,
                 loss_fn=None):

        super(REDNet, self).__init__()

        self.loss_fn = loss_fn
        self.norm_layer = norm_layer
        self._up_kwargs = up_kwargs
        self.nclass = nclass

        if backbone == 'resnet18':
            self.expansion = 1
            self.resnet = Resnet18(norm_layer=norm_layer)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        # bilinear upsample options

        #fix Atention

        self.at_16 = Attention_Mod(256, norm_layer=norm_layer)
        self.at_8 = Attention_Mod(128, norm_layer=norm_layer)

        self.atlayer_16 = FPNOutput(256, 128, 128,norm_layer=norm_layer)

        self.clslayer_16 = FPNOutput(128, 64, nclass,norm_layer=norm_layer)
        self.clslayer_8 = FPNOutput(128, 64, nclass,norm_layer=norm_layer)    
        self.clslayer_output  = FPNOutput(256, 256, nclass,norm_layer=norm_layer)

    def forward(self, x, lbl=None):


        _, _, h, w = x.size()


        #feat 16 - 3stage : 3x3x256
        #feat 8  - 2stage : 3x3x128

        feat4, feat8, feat16, feat32 = self.resnet(x)
        
        #print(feat8.size())  # 4x128x24x24
        #print(feat16.size()) # 4x256x12x12


        atout_8 = self.at_8(feat8)

        atout_16 = self.at_16(feat16)
        atout_16 = self.atlayer_16(atout_16)

        #print(atout_8.size())  # 4x128x24x24
        #print(atout_16.size()) # 4x128x12x12


        x = self._upsample_cat(atout_8, atout_16)
        #print(x.size()) # 4x256x12x12

        x = self.clslayer_output(x)
        #print(x.size())

        outputs = F.interpolate(x, (h,w), **self._up_kwargs)
        #print(outputs.size())
        # assert False
        
        if self.training:
            auxout_1 = self.clslayer_8 (atout_8)
            auxout_2 = self.clslayer_16 (atout_16)
            auxout_1 = F.interpolate(auxout_1, (h,w), **self._up_kwargs)
            auxout_2 = F.interpolate(auxout_2, (h,w), **self._up_kwargs)
            loss = self.loss_fn(outputs,lbl)+0.5*self.loss_fn(auxout_1,lbl)+0.5*self.loss_fn(auxout_2,lbl)
            return loss
        else:
            return outputs



    def _upsample_cat(self, x1, x2):
        '''Upsample and concatenate feature maps.
        '''
        _,_,H,W = x2.size()
        x1 = F.interpolate(x1, (H,W), **self._up_kwargs)
        x = torch.cat([x1,x2],dim=1)
        return x

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            if isinstance(child,(OhemCELoss2D,CrossEntropyLoss)):
                continue
            elif isinstance(child, (Attention_Mod, FPNOutput)):
                child_wd_params, child_nowd_params = child.get_params()
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:

                child_wd_params, child_nowd_params = child.get_params()
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params



class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, norm_layer=None, activation='leaky_relu',*args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.norm_layer = norm_layer
        if self.norm_layer is not None:
            self.bn = norm_layer(out_chan, activation=activation)
        else:
            self.bn =  lambda x:x

        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)



class FPNOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, norm_layer=None, *args, **kwargs):
        super(FPNOutput, self).__init__()
        self.norm_layer = norm_layer
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1, norm_layer=norm_layer)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, self.norm_layer):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params



class Attention_Mod(nn.Module):
    """ Attention Module """
    def __init__(self, in_dim,
                 norm_layer=None):
        super(Attention_Mod, self).__init__()
        self.channel_in = in_dim
        self.norm_layer = norm_layer

        self.query_conv = ConvBNReLU(in_chan=in_dim, out_chan=in_dim//8, ks=1, padding=0,norm_layer=norm_layer, activation='none')
        self.key_conv = ConvBNReLU(in_chan=in_dim, out_chan=in_dim//8,  ks=1, padding=0, norm_layer=norm_layer, activation='none')
        self.value_conv = ConvBNReLU(in_chan=in_dim, out_chan=in_dim, ks=1, padding=0, norm_layer=norm_layer)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        self.init_weight()


    def forward(self, x):
        """
            input: 
                x : input feature maps (B x C x H x W)
            returns:
                out : attention value + input feature
                attention: B x (HxW) x (HxW)
        """

        m_batchsize, m_channel, m_height, m_width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, m_width*m_height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, m_width*m_height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, m_width*m_height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, m_channel, m_height, m_width)

        out = self.gamma*out + x
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, self.norm_layer):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params




