import torch as t

class QuanConv2d(t.nn.Conv2d):
    def __init__(self, m: t.nn.Conv2d, quan_w_fn=None, quan_a_fn=None, act_init=None, progress= False):
        assert type(m) == t.nn.Conv2d
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
                         stride=m.stride,
                         padding=m.padding,
                         dilation=m.dilation,
                         groups=m.groups,
                         bias=True if m.bias is not None else False,
                         padding_mode=m.padding_mode)
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn

        if progress is False:
            #load params from before model
            self.weight = t.nn.Parameter(m.weight.detach())
            if m.bias is not None:
                self.bias = t.nn.Parameter(m.bias.detach())
            #initialize scale
            self.quan_w_fn.init_from(m.weight)
            if act_init is not None:
                self.quan_a_fn.init_from(act_init)
    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        quantized_act = self.quan_a_fn(x)
        return self._conv_forward(quantized_act, quantized_weight)


class QuanLinear(t.nn.Linear):
    def __init__(self, m: t.nn.Linear, quan_w_fn=None, quan_a_fn=None, act_init=None, progress= False):
        assert type(m) == t.nn.Linear
        super().__init__(m.in_features, m.out_features,
                         bias=True if m.bias is not None else False)
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn
        if progress is False:
            #load params from before model
            self.weight = t.nn.Parameter(m.weight.detach())
            if m.bias is not None:
                self.bias = t.nn.Parameter(m.bias.detach())
            #initialize scale
            self.quan_w_fn.init_from(m.weight)
            if act_init is not None:
                self.quan_a_fn.init_from(act_init)
    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        quantized_act = self.quan_a_fn(x)
        return t.nn.functional.linear(quantized_act, quantized_weight, self.bias)


QuanModuleMapping = {
    t.nn.Conv2d: QuanConv2d,
    t.nn.Linear: QuanLinear,
    QuanConv2d: QuanConv2d,
    QuanLinear: QuanLinear
}




class LsqQuantizer(t.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, scale_grad, thd_neg, thd_pos, act_mode = True):
        # scale grad, thd_neg, thd_pos is scaler else tensor
        x_d_s = t.divide(input /scale)
        x_ = t.round(t.clamp( x_d_s, thd_neg, thd_pos))
        output = x_*scale

        ctx.save_for_backward(x_d_s, x_, scale_grad, thd_neg, thd_pos, act_mode)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x_d_s, x_, scale_grad, thd_neg, thd_pos, act_mode = ctx.saved_tensors
        
        STE = grad_output.clone()
        if act_mode is True:
            STE[x_d_s < thd_neg] = 0
            STE[x_d_s > thd_pos] = 0

        s_grad = x_
        if x_d_s < thd_pos and x_d_s > thd_neg:
            s_grad += -x_d_s * scale_grad

        
        return STE, s_grad, None, None, None



class QuanConv2d(t.nn.Conv2d):
    def __init__(self, m: t.nn.Conv2d, quan_w_fn=None, quan_a_fn=None, act_init=None, progress= False):
        assert type(m) == t.nn.Conv2d
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
                         stride=m.stride,
                         padding=m.padding,
                         dilation=m.dilation,
                         groups=m.groups,
                         bias=True if m.bias is not None else False,
                         padding_mode=m.padding_mode)
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn

        if progress is False:
            #load params from before model
            self.weight = t.nn.Parameter(m.weight.detach())
            if m.bias is not None:
                self.bias = t.nn.Parameter(m.bias.detach())
            #initialize scale
            self.quan_w_fn.init_from(m.weight)
            if act_init is not None:
                self.quan_a_fn.init_from(act_init)
    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        quantized_act = self.quan_a_fn(x)
        return self._conv_forward(quantized_act, quantized_weight)

        
class LsqQuan(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=False, activation = True):
        super().__init__(bit)
        self.thd_neg = None
        self.thd_pos = None
        self.activation = activation

        self.set_thd(bit, all_positive, symmetric)
        self.per_channel = per_channel
        self.s = t.nn.Parameter(t.ones(1))

    def set_thd(self, bit, all_positive=False, symmetric=False):
        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                print(bit)
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1
        # print(f'set bit as : {bit}')

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            self.s = t.nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
        else:
            self.s = t.nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))

    def forward(self, x):
        # if self.per_channel:
        #     s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        if self.activation:
            s_grad_scale = 0.1 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 0.0001 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(self.s, s_grad_scale)

        x = x / s_scale
        x = t.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        x = x * (s_scale.detach())
        return x
