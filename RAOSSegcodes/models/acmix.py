import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
# 修改
def position(D, H, W, is_cuda=False):
    if is_cuda:
        loc_d = torch.linspace(-1.0, 1.0, D).cuda().unsqueeze(1).unsqueeze(2).repeat(1, W, H)
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(1).unsqueeze(0).repeat(D, 1, H)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(0).unsqueeze(0).repeat(D, W, 1)
    else:
        loc_d = torch.linspace(-1.0, 1.0, D).unsqueeze(1).unsqueeze(2).repeat(1, W, H)
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(1).unsqueeze(0).repeat(D, 1, H)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(0).unsqueeze(0).repeat(D, W, 1)
    loc = torch.cat([loc_d.unsqueeze(0),loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc

# 寻找离一个数平方根最近的因子
def closest_factor_to_sqrt(n):
    if n <= 1:
        return n  # 0 and 1 are their own factors
    sqrt_n = int(math.sqrt(n))
    # Check if sqrt_n is a perfect square root (i.e., sqrt_n * sqrt_n == n)
    if sqrt_n * sqrt_n == n:
        return sqrt_n
        # Search for the closest factor starting from sqrt_n going downwards
    closest_factor = float('inf')
    closest_diff = float('inf')
    for i in range(sqrt_n, 0, -1):
        if n % i == 0:
            factor_diff = abs(i - (n // i))
            if factor_diff < closest_diff:
                closest_diff = factor_diff
                closest_factor = i
    return closest_factor

def stride(x, stride):
    b, c, d, h, w = x.shape
    return x[:, :, ::stride, ::stride]


def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)


class ACmix(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
        super(ACmix, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.head_dim = self.out_planes // self.head

        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv3d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv3d(in_planes, out_planes, kernel_size=1)
        self.conv_p = nn.Conv3d(3, self.head_dim, kernel_size=1)

        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = torch.nn.Softmax(dim=1)

        self.fc = nn.Conv3d(3 * self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)
        self.dep_conv = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, out_planes,
                                  kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=1,
                                  stride=stride)
        self.reset_parameters()

    def reset_parameters(self):
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i // self.kernel_conv, i % self.kernel_conv] = 1.
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)

    def forward(self, x):
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)#  q:4×32×16×64×64
        scaling = float(self.head_dim) ** -0.5
        b, c, d, w, h = q.shape
        d_out, h_out, w_out =d // self.stride, h // self.stride, w // self.stride
        # ### att
        # ## positional encoding

        factor_h = closest_factor_to_sqrt(d)  # 3d到2d的一个转换因子
        factor_w = int(d / factor_h)

        pe = self.conv_p(position(d, h, w, x.is_cuda)).view(-1 ,self.head_dim,factor_h * h, factor_w * w) # pe:1×8×16×64×64 -> 1*8*256*256
        q_att = (q.view(b * self.head, self.head_dim, d, h, w) * scaling).view(b * self.head, self.head_dim, factor_h * h,factor_w * w)
        # q_att:16×8×16×64×64 -> 16*8*256*256
        k_att = k.view(b * self.head, self.head_dim, d, h, w).view(b * self.head, self.head_dim, factor_h * h,factor_w * w)
        v_att = v.view(b * self.head, self.head_dim, d, h, w).view(b * self.head, self.head_dim, factor_h * h,factor_w * w)

        if self.stride > 1:
            q_att = stride(q_att, self.stride)
            q_pe = stride(pe, self.stride)
        else:
            q_pe = pe

        unfold_k = self.unfold(self.pad_att(k_att)).view(b * self.head, self.head_dim,
                                                         self.kernel_att * self.kernel_att, factor_h * h_out,
                                                         factor_w * w_out)  # b*head, head_dim, k_att^2, h_out, w_out 16*8*49*256*256

        unfold_rpe = self.unfold(self.pad_att(pe)).view(1, self.head_dim, self.kernel_att * self.kernel_att, factor_h * h_out,
                                                       factor_w * w_out)  # 1, head_dim, k_att^2, h_out, w_out 1*8*49*256*256

        att = (q_att.unsqueeze(2) * (unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(1)  # (b*head, head_dim, 1, h_out, w_out) * (b*head, head_dim, k_att^2, h_out, w_out) -> (b*head, k_att^2, h_out, w_out)
        att = self.softmax(att)

        out_att = self.unfold(self.pad_att(v_att)).view(b * self.head, self.head_dim, self.kernel_att * self.kernel_att,
                                                        factor_h * h_out, factor_w * w_out)
        out_att = (att.unsqueeze(1) * out_att).sum(2).view(b, self.out_planes,-1,w_out,h_out)# 2维数据转成3维数据

        ## conv
        f_all = self.fc(torch.cat(
            [q.view(b, self.head, self.head_dim, d * h * w), k.view(b, self.head, self.head_dim, d * h * w),
             v.view(b, self.head, self.head_dim, d * h * w)], 1).permute(1, 0, 2, 3))
        f_conv = f_all.permute(1, 0, 2, 3).reshape(b, -1 , d, h * w)
        out_conv = self.dep_conv(f_conv).reshape(b, -1, d, w, h)
        return self.rate1 * out_att + self.rate2 * out_conv

if __name__=='__main__':
    x = torch.rand([2, 128, 16, 16, 20])
    net = ACmix(in_planes=128, out_planes=256)
    print(net(x).shape)

    # p = position(10,64,64,is_cuda=False)
    # print(p.shape)
