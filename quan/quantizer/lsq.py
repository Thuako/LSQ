import torch as t

from .quantizer import Quantizer

class LsqQuan(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True, activation = True, name=''):
        super().__init__(bit)
        self.thd_neg = None
        self.thd_pos = None
        self.activation = activation
        self.s = None
        self.set_thd(bit, all_positive, symmetric)
        self.per_channel = per_channel
        # self.s = t.nn.Parameter(t.ones(1))
        self.s_grad_scale = None
        self.name = 'not setted init'

    def set_thd(self, bit, all_positive=False, symmetric=False):
        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            target_thd_neg = 0
            target_thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                target_thd_neg = - 2 ** (bit - 1) + 1
                target_thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                target_thd_neg = - 2 ** (bit - 1)
                target_thd_pos = 2 ** (bit - 1) - 1
        
        if self.s is not None:
            # print(self.s)
            self.s = t.nn.Parameter((self.s * (self.thd_pos ** 0.5)) / target_thd_pos ** 0.5)
        self.thd_neg = target_thd_neg
        self.thd_pos = target_thd_pos
        # print(f'set bit as : {bit}')

    def init_from(self, x, activation, conv=True, Netsize=None, per_layer=False, name=''):
        self.name = name
        self.activation = activation
        if Netsize is not None:
            self.Netsize = Netsize
        if per_layer is True:
            if conv is True:
                if activation is False:
                    self.s = t.nn.Parameter( t.ones(x.size()) * x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5)) 
                elif activation is True:
                    self.s = t.nn.Parameter( t.ones( x.size()[-1:] ) * x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5)  )
            else:
                if activation is False:
                    self.s = t.nn.Parameter( t.ones( x.size() ) * x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))
                elif activation is True:
                    self.s = t.nn.Parameter(  t.ones( x.size()[-1:] ) * x.detach().abs().mean(dim=0, keepdim=True).squeeze(0) * 2 / (self.thd_pos ** 0.5))
        else:
            self.s = t.nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))
        
        if activation is True:
            self.s_grad_scale = 1 / ((self.thd_pos ) ** 0.5)
        else:
            self.s_grad_scale = 1 / ((self.thd_pos ) ** 0.5)



    def forward(self, x):
        # print(f'==========={self.name}=========')
        if self.activation is True:
            return LsqQuantizer_A.apply(x, self.s, t.tensor(self.s_grad_scale), t.tensor(self.thd_neg), t.tensor(self.thd_pos) )
        else:
            return LsqQuantizer_W.apply(x, self.s, t.tensor(self.s_grad_scale), t.tensor(self.thd_neg), t.tensor(self.thd_pos) )
        

class LsqQuantizer_A(t.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, scale_grad, thd_neg, thd_pos):
        # print('forward is working')
        # scale grad, thd_neg, thd_pos is scaler else tensor
        x_d_s = t.div(input, scale)
        x_ = t.round(t.clamp( x_d_s, thd_neg, thd_pos))
        output = x_ * scale
        # print(f'x_d_s : {x_d_s.size()}, x_ : {x_.size()}, output : {output.size()}')
        ctx.save_for_backward(x_d_s, x_, scale_grad, thd_neg, thd_pos)
        return output 

    @staticmethod
    def backward(ctx, grad_output):
        x_d_s, x_, scale_grad, thd_neg, thd_pos = ctx.saved_tensors        
        diff_htanh_mask = 1 - ((x_d_s > thd_pos ) + (x_d_s < thd_neg)).float()
        STE = grad_output.clone()

        # print(STE.size(), diff_htanh_mask.size(), (STE * diff_htanh_mask).size())
        STE = STE * diff_htanh_mask

        s_grad = (x_ - x_d_s * diff_htanh_mask )

        return STE, grad_output * s_grad *0.1* scale_grad, None, None, None

class LsqQuantizer_W(t.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, scale_grad, thd_neg, thd_pos):
        # print('forward is working')
        # scale grad, thd_neg, thd_pos is scaler else tensor
        x_d_s = t.div(input, scale)
        x_ = t.round(t.clamp( x_d_s, thd_neg, thd_pos))
        output = x_ * scale
        # print(f'x_d_s : {x_d_s.size()}, x_ : {x_.size()}, output : {output.size()}')
        ctx.save_for_backward(x_d_s, x_, scale_grad, thd_neg, thd_pos)
        return output 

    @staticmethod
    def backward(ctx, grad_output):
        x_d_s, x_, scale_grad, thd_neg, thd_pos = ctx.saved_tensors        
        diff_htanh_mask = 1 - ((x_d_s > thd_pos ) + (x_d_s < thd_neg)).float()
        
        STE = grad_output.clone()

        s_grad = (x_ - x_d_s * diff_htanh_mask )
        # print(s_grad, STE.reshape(-1)[0])

        return STE, grad_output * s_grad *0.0001* scale_grad, None, None, None


class LsqQuantizer_A_1(t.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, scale_grad, thd_neg, thd_pos):
        # print('forward is working')
        # scale grad, thd_neg, thd_pos is scaler else tensor
        x_d_s = t.div(input, scale)
        x_ = t.round(t.clamp( x_d_s, thd_neg, thd_pos))
        output = x_ * scale
        # print(f'x_d_s : {x_d_s.size()}, x_ : {x_.size()}, output : {output.size()}')
        ctx.save_for_backward(x_d_s, x_, scale, scale_grad, thd_neg, thd_pos)
        return output 

    @staticmethod
    def backward(ctx, grad_output):
        x_d_s, x_, scale, scale_grad, thd_neg, thd_pos = ctx.saved_tensors        
        diff_htanh_mask = 1 - ((x_d_s > thd_pos ) + (x_d_s < 0)).float()
        STE = grad_output.clone()

        # print(STE.size(), diff_htanh_mask.size(), (STE * diff_htanh_mask).size())
        STE = STE * diff_htanh_mask

        s_grad = (t.clamp( x_, min=0) - x_d_s * diff_htanh_mask   ) * scale_grad

        # print(s_grad)

        return STE, s_grad , None, None, None

class LsqQuantizer_W_channel(t.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, scale_grad, thd_neg, thd_pos):
        # print('forward is working')
        # scale grad, thd_neg, thd_pos is scaler else tensor
        x_d_s = t.div(input, scale)
        x_ = t.round(t.clamp( x_d_s, thd_neg, thd_pos))
        output = x_ * scale
        # print(f'x_d_s : {x_d_s.size()}, x_ : {x_.size()}, output : {output.size()}')
        ctx.save_for_backward(x_d_s, x_, scale, scale_grad, thd_neg, thd_pos)
        return output 

    @staticmethod
    def backward(ctx, grad_output):
        x_d_s, x_, scale, scale_grad, thd_neg, thd_pos = ctx.saved_tensors        
        diff_htanh_mask = 1 - ((x_d_s > thd_pos ) + (x_d_s < 0)).float()
        
        STE = grad_output.clone()

        s_grad = ((t.clamp( x_, min=0) - x_d_s * diff_htanh_mask  ) * scale_grad)
        # print(s_grad, STE.reshape(-1)[0])

        print(s_grad.size())

        return STE, s_grad* 0.0001, None, None, None


    # def set_thd(self, bit, all_positive=False, symmetric=False):
    #     if all_positive:
    #         assert not symmetric, "Positive quantization cannot be symmetric"
    #         # unsigned activation is quantized to [0, 2^b-1]
    #         self.thd_neg = 0
    #         self.thd_pos = 2 ** bit - 1
    #     else:
    #         if symmetric:
    #             # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
    #             self.thd_neg = - 2 ** (bit - 1) + 1
    #             self.thd_pos = 2 ** (bit - 1) - 1
    #         else:
    #             # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
    #             self.thd_neg = - 2 ** (bit - 1)
    #             self.thd_pos = 2 ** (bit - 1) - 1        




