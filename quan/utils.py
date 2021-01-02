import logging
import math
from .func import *
from .quantizer import *

logger = logging.getLogger()

def quantizer(default_cfg, this_cfg=None, before_model=None, activation=False):
    target_cfg = dict(default_cfg)
    if this_cfg is not None:
        for k, v in this_cfg.items():
            target_cfg[k] = v

    if target_cfg['bit'] is None:
        q = IdentityQuan
    elif target_cfg['mode'] == 'lsq':
        q = LsqQuan
    else:
        raise ValueError('Cannot find quantizer `%s`', target_cfg['mode'])

    target_cfg.pop('mode')
    target_cfg['activation'] = activation
    if before_model == None:
        return q(**target_cfg)
    else:
        return q()


def find_modules_to_quantize(model, args):
    replaced_modules = dict()
    quan_scheduler = args[0]
    act_init = args[1]
    #     act_init = kargs['activation']
    # quan_scheduler = kargs['args_quan']
    for name, module in model.named_modules():
        # print(f'%%%%%%%%%%%%%%\n\n name : {name}\nmodule : {module}\n\n%%%%%%%%%%%%%%')
        # type(module), .keys()는 스트링 아님 --> if in 이 정확해야 True 반환
        if type(module) in QuanModuleMapping.keys():
            if name in quan_scheduler.excepts:
                # print(f'model.named_modules() : {name}, type(module) : {type(module)}')
                replaced_modules[name] = QuanModuleMapping[type(module)](
                    module,
                    # quan_w_fn = LsqQaun or IdentityQuan
                    quan_w_fn=quantizer(quan_scheduler.weight,
                                        quan_scheduler.excepts[name].weight, activation=False),
                    quan_a_fn=quantizer(quan_scheduler.act,
                                        quan_scheduler.excepts[name].act, activation=True),
                    act_init=act_init[name] if act_init is not None else None,
                    per_layer=False,
                    name=str(name)
                )
                # print(act_init[name].size())
            else:
                # print(f'model.named_modules() : {name}, type(module) : {type(module)}')
                replaced_modules[name] = QuanModuleMapping[type(module)](
                    module,
                    quan_w_fn=quantizer(quan_scheduler.weight, activation=False),
                    quan_a_fn=quantizer(quan_scheduler.act, activation=True),
                    act_init=act_init[name] if act_init is not None else None,
                    per_layer=False,
                    name=str(name)
                )
                # print(act_init[name].size())
        elif name in quan_scheduler.excepts:
            logging.warning('Cannot find module %s in the model, skip it' % name)
    return replaced_modules


def replace_module_by_names(model, modules_to_replace):
    def helper(child: t.nn.Module):
        for n, c in child.named_children():
            if type(c) in QuanModuleMapping.keys():
                for full_name, m in model.named_modules():
                    if c is m:
                        tmp = modules_to_replace.pop(full_name)
                        child.add_module(n, tmp)
                        # print(n, tmp)
                        break
            else:
                helper(c)
    helper(model)
    return model



def change_bit_width(model, args=None):
    quan_scheduler = args[0]
    act_init = args[1] if args[1] is not None else None
    for name, module in model.named_modules():
        if type(module) in QuanModuleMapping.keys():
            if name in quan_scheduler.excepts:
                module.quan_w_fn.set_thd(**dict(quan_scheduler.excepts[name].weight))
                module.quan_a_fn.set_thd(**dict(quan_scheduler.excepts[name].act))
                if act_init is not None:
                    module.quan_w_fn.init_from(module.weight)
                    module.quan_a_fn.init_from(act_init[name])

                w_bit = int(math.log(module.quan_w_fn.thd_pos + 1, 2))
                a_bit = int(math.log(module.quan_a_fn.thd_pos + 1, 2))

                logger.info(f'{name} layer bit set by \
                    weight:{quan_scheduler.excepts[name].weight}bit   act :{quan_scheduler.excepts[name].act}bit')
                logger.info('changed_bit_with func ' + str(name) + 'layer ##### weight pos_thd : ' + str(w_bit) + 'act pos_thd : ' + str(a_bit))

            else:
                module.quan_w_fn.set_thd(**dict(quan_scheduler.weight))
                module.quan_a_fn.set_thd(**dict(quan_scheduler.act))
                if act_init is not None:
                    module.quan_w_fn.init_from(module.weight)
                    module.quan_a_fn.init_from(act_init[name])
                w_bit = int(math.log(module.quan_w_fn.thd_pos + 1, 2))
                a_bit = int(math.log(module.quan_a_fn.thd_pos + 1, 2))

                logging.info(f'{name} layer bit changed by \
                        weight:{quan_scheduler.weight.bit}bit   act :{quan_scheduler.act.bit}bit')                
        elif name in quan_scheduler.excepts:
            logging.warning('Cannot find module %s in the model, skip it' % name)
    
    check_changed_bit(model)
    return model

def check_changed_bit(model):
    for name, module in model.named_modules():
        if type(module) in QuanModuleMapping.keys():
            w_bit = int(math.log(module.quan_w_fn.thd_pos + 1, 2))
            a_bit = int(math.log(module.quan_a_fn.thd_pos + 1, 2))
            msg = str(name) + 'layer ##### weight pos_thd : ' + str(w_bit) + '    act pos_thd : ' + str(a_bit)
            logger.info(msg)
    