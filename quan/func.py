import torch as t

class QuanConv2d(t.nn.Conv2d):
    def __init__(self, m: t.nn.Conv2d, quan_w_fn=None, quan_a_fn=None, act_init=None, progress= False, per_layer=False, name=''):
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
        self.name= name
        
        if per_layer is False:
            self.Netsize_A = m.weight.size().numel()
            self.Netsize_W = m.weight.size().numel() * m.weight.size(-2) * m.weight.size(-1) / m.weight.size(0)
        else:
            self.Netsize_A = m.weight.size().numel() / m.weight.size(0)
            self.Netsize_W = m.weight.size().numel() * m.weight.size(-2) * m.weight.size(-1) / m.weight.size(0)

        if progress is False:
            #load params from before model
            self.weight = t.nn.Parameter(m.weight.detach())
            if m.bias is not None:
                self.bias = t.nn.Parameter(m.bias.detach())
            #initialize scale
            self.quan_w_fn.init_from(m.weight, activation=False, conv=True, Netsize= self.Netsize_W, name=self.name + '_quan_W')
            if act_init is not None:
                self.quan_a_fn.init_from(act_init, activation= True, conv=True, Netsize= self.Netsize_A, name= self.name + '_quan_A')
    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        quantized_act = self.quan_a_fn(x)
        return self._conv_forward(quantized_act, quantized_weight)


class QuanLinear(t.nn.Linear):
    def __init__(self, m: t.nn.Linear, quan_w_fn=None, quan_a_fn=None, act_init=None, progress= False, per_layer= False, name=''):
        assert type(m) == t.nn.Linear
        super().__init__(m.in_features, m.out_features,
                         bias=True if m.bias is not None else False)
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn
        self.name = name

        if per_layer is False:
            self.Netsize_A = m.weight.size().numel()
            self.Netsize_W = m.weight.size().numel()
        else:
            self.Netsize_A = m.weight.size(0)
            self.Netsize_W = 1

        if progress is False:
            #load params from before model
            self.weight = t.nn.Parameter(m.weight.detach())
            if m.bias is not None:
                self.bias = t.nn.Parameter(m.bias.detach())
            #initialize scale
            self.quan_w_fn.init_from(m.weight, activation=False, conv=False, Netsize= self.Netsize_W, per_layer=per_layer, name = self.name + '_quan_W')
            if act_init is not None:
                self.quan_a_fn.init_from(act_init, activation=True, conv=False, Netsize=self.Netsize_A, per_layer=per_layer, name = self.name + '_quan_A')
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

# QuanModuleMapping = {
#     t.nn.Conv2d: QuanConv2d,
#     t.nn.Linear: QuanLinear,
#     QuanConv2d: QuanConv2d_progress,
#     QuanLinear: QuanLinear_progress
# }

# class QuanConv2d_progress(QuanConv2d):
#     def __init__(self, m: QuanConv2d, quan_w_fn=None, quan_a_fn=None, act_init = None):
#         assert type(m) == QuanConv2d
#         Parent = t.nn.Conv2d(m.in_channels, m.out_channels, m.kernel_size,
#                          stride=m.stride,
#                          padding=m.padding,
#                          dilation=m.dilation,
#                          groups=m.groups,
#                          bias=True if m.bias is not None else False,
#                          padding_mode=m.padding_mode)
#         super().__init__(Parent, progress = True)
#         self.quan_w_fn = quan_w_fn
#         self.quan_a_fn = quan_a_fn

#         #load params from before model
#         self.weight = t.nn.Parameter(m.weight.detach())
#         if m.bias is not None:
#             self.bias = t.nn.Parameter(m.bias.detach())
#         # initialize new scale
#         if act_init is not None:
#             self.quan_w_fn.init_from(m.weight)
#             self.quan_a_fn.init_from(act_init)
#         else: # use before quantizationd model's scale
#             self.quan_w_fn.s = m.quan_w_fn.s
#             self.quan_a_fn.s = m.quan_a_fn.s
#     def forward(self, x):
#         quantized_weight = self.quan_w_fn(self.weight)
#         quantized_act = self.quan_a_fn(x)
#         return self._conv_forward(quantized_act, quantized_weight)


# class QuanLinear_progress(QuanLinear):
#     def __init__(self, m: QuanLinear, quan_w_fn=None, quan_a_fn=None, act_init=None):
#         assert type(m) == QuanLinear
#         Parent = t.nn.Linear( m.in_features, m.out_features, bias=True if m.bias is not None else False)
#         super().__init__(Parent, progress = True)
#         self.quan_w_fn = quan_w_fn
#         self.quan_a_fn = quan_a_fn
        
#         #load params from before model
#         self.weight = t.nn.Parameter(m.weight.detach())
#         if m.bias is not None:
#             self.bias = t.nn.Parameter(m.bias.detach())
        
#         # initialize new scale
#         if act_init is not None:
#             self.quan_w_fn.init_from(m.weight)
#             self.quan_a_fn.init_from(act_init)
#         else: # use before quantizationd model's scale
#             self.quan_w_fn.s = m.quan_w_fn.s
#             self.quan_a_fn.s = m.quan_a_fn.s
#     def forward(self, x):
#         quantized_weight = self.quan_w_fn(self.weight)
#         quantized_act = self.quan_a_fn(x)
#         return t.nn.functional.linear(quantized_act, quantized_weight, self.bias)

