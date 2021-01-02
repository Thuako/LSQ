import logging
from pathlib import Path

import torch as t
import yaml

import process
import quan
import util
from model import create_model
import copy

import argparse
import munch
import yaml



def main(args):
    script_dir = Path.cwd()
    # args = util.get_config(default_file=script_dir / 'config.yaml')
    output_dir = script_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)

    log_dir = util.init_logger(args.name, output_dir, 'logging.conf')
    logger = logging.getLogger()

    with open(log_dir / "args.yaml", "w") as yaml_file:  # dump experiment config
        yaml.safe_dump(args, yaml_file)

    pymonitor = util.ProgressMonitor(logger)
    tbmonitor = util.TensorBoardMonitor(logger, log_dir)
    monitors = [pymonitor , tbmonitor]

    if args.device.type == 'cpu' or not t.cuda.is_available() or args.device.gpu == []:
        args.device.gpu = []
    else:
        available_gpu = t.cuda.device_count()
        for dev_id in args.device.gpu:
            if dev_id >= available_gpu:
                logger.error('GPU device ID {0} requested, but only {1} devices available'
                             .format(dev_id, available_gpu))
                exit(1)
        # Set default device in case the first one on the list
        t.cuda.set_device(args.device.gpu[0])
        # Enable the cudnn built-in auto-tuner to accelerating training, but it
        # will introduce some fluctuations in a narrow range.
        t.backends.cudnn.benchmark = True
        t.backends.cudnn.deterministic = False

    # Initialize data loader
    train_loader, val_loader, test_loader = util.load_data(args.dataloader)
    logger.info('Dataset `%s` size:' % args.dataloader.dataset +
                '\n          Training Set = %d (%d)' % (len(train_loader.sampler), len(train_loader)) +
                '\n        Validation Set = %d (%d)' % (len(val_loader.sampler), len(val_loader)) +
                '\n              Test Set = %d (%d)' % (len(test_loader.sampler), len(test_loader)))
    
    # Define loss function (criterion) and optimizer
    criterion = t.nn.CrossEntropyLoss().to(args.device.type)

    # Create the model ( download )
    model = create_model(args)
    
    # load activation each layer to initialize activation quantize scale
    act_init = process.get_initialization(test_loader, model, criterion, -2, monitors, args)
    # replace model
    modules_to_replace = quan.find_modules_to_quantize(model, [args.quan, act_init])


    model = quan.replace_module_by_names(model, modules_to_replace)
    logger.info('Inserted quantizers into the original model')
    
    # print new model
    # for name, module in model.named_modules():
    #     print(name, '\t', type(module))

    # progressive quantization mode
    if args.quan.progress.mode is True:
        model, _, _= util.load_checkpoint(model, args.quan.progress.path, args.device.type, lean=args.resume.lean)        
        progress_act_init = process.get_initialization(test_loader, model, criterion, -2, monitors, args) if args.quan.progress.init_restart else None
        model = quan.change_bit_width(model, [args.quan.progress, progress_act_init])
        quan.check_changed_bit(model)
    
    model.to('cpu')
    tbmonitor.writer.add_graph(model, input_to_model=train_loader.dataset[0][0].unsqueeze(0))



    # resume 
    start_epoch = 0
    if args.resume.path:
        model, start_epoch, _ = util.load_checkpoint(
            model, args.resume.path, args.device.type, lean=args.resume.lean)
        
    # data Parallel by new model
    if args.device.gpu and not args.dataloader.serialized:
        model = t.nn.DataParallel(model, device_ids=args.device.gpu)

    model.to(args.device.type)
    
    # optimizer = t.optim.Adam(model.parameters(), lr=args.optimizer.learning_rate)
    optimizer = t.optim.SGD(model.parameters(),
                            lr=args.optimizer.learning_rate,
                            momentum=args.optimizer.momentum,
                            weight_decay=args.optimizer.weight_decay)
    lr_scheduler = util.lr_scheduler(optimizer,
                                     batch_size=train_loader.batch_size,
                                     num_samples=len(train_loader.sampler),
                                     **args.lr_scheduler)
    logger.info(('Optimizer: %s' % optimizer).replace('\n', '\n' + ' ' * 11))
    logger.info('LR scheduler: %s\n' % lr_scheduler)

    print(f'**********train_loader_sampler : {len(train_loader.sampler)}')

    perf_scoreboard = process.PerformanceScoreboard(args.log.num_best_scores)

    if args.eval:
        process.validate(test_loader, model, criterion, -1, monitors, args)
    else:  # training
        if args.resume.path or args.pre_trained:
            logger.info('>>>>>>>> Epoch -1 (pre-trained model evaluation)')
            top1, top5, _ = process.validate(val_loader, model, criterion,
                                             start_epoch - 1, monitors, args)
            perf_scoreboard.update(top1, top5, start_epoch - 1)
        for epoch in range(start_epoch, args.epochs):
            logger.info('>>>>>>>> Epoch %3d' % epoch)
            t_top1, t_top5, t_loss = process.train(train_loader, model, criterion, optimizer,
                                                   lr_scheduler, epoch, monitors, args)
            v_top1, v_top5, v_loss = process.validate(val_loader, model, criterion, epoch, monitors, args)

            tbmonitor.writer.add_scalars('Train_vs_Validation/Loss', {'train': t_loss, 'val': v_loss}, epoch)
            tbmonitor.writer.add_scalars('Train_vs_Validation/Top1', {'train': t_top1, 'val': v_top1}, epoch)
            tbmonitor.writer.add_scalars('Train_vs_Validation/Top5', {'train': t_top5, 'val': v_top5}, epoch)

            perf_scoreboard.update(v_top1, v_top5, epoch)
            is_best = perf_scoreboard.is_best(epoch)
            util.save_checkpoint(epoch, args.arch, model, {'top1': v_top1, 'top5': v_top5}, is_best, args.name, log_dir, args.dataloader.serialized)

        logger.info('>>>>>>>> Epoch -1 (final model evaluation)')
        process.validate(test_loader, model, criterion, -1, monitors, args)
    
    
    if args.save_all is True:
        filename = 'raw_model.pth'
        filepath = os.path.join(output_dir, filename)

    tbmonitor.writer.close()  # close the TensorBoard
    logger.info('Program completed successfully ... exiting ...')
    logger.info('If you have any questions or suggestions, please visit: github.com/zhutmost/lsq-net')


def train_full_precision(args):
    script_dir = Path.cwd()
    # args = util.get_config(default_file=script_dir / 'config.yaml')
    output_dir = script_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)

    log_dir = util.init_logger(args.name, output_dir, 'logging.conf')
    logger = logging.getLogger()

    with open(log_dir / "args.yaml", "w") as yaml_file:  # dump experiment config
        yaml.safe_dump(args, yaml_file)

    pymonitor = util.ProgressMonitor(logger)
    tbmonitor = util.TensorBoardMonitor(logger, log_dir)
    monitors = [pymonitor, tbmonitor]

    if args.device.type == 'cpu' or not t.cuda.is_available() or args.device.gpu == []:
        args.device.gpu = []
    else:
        available_gpu = t.cuda.device_count()
        for dev_id in args.device.gpu:
            if dev_id >= available_gpu:
                logger.error('GPU device ID {0} requested, but only {1} devices available'
                             .format(dev_id, available_gpu))
                exit(1)
        # Set default device in case the first one on the list
        t.cuda.set_device(args.device.gpu[0])
        # Enable the cudnn built-in auto-tuner to accelerating training, but it
        # will introduce some fluctuations in a narrow range.
        t.backends.cudnn.benchmark = True
        t.backends.cudnn.deterministic = False

    # Initialize data loader
    train_loader, val_loader, test_loader = util.load_data(args.dataloader)
    logger.info('Dataset `%s` size:' % args.dataloader.dataset +
                '\n          Training Set = %d (%d)' % (len(train_loader.sampler), len(train_loader)) +
                '\n        Validation Set = %d (%d)' % (len(val_loader.sampler), len(val_loader)) +
                '\n              Test Set = %d (%d)' % (len(test_loader.sampler), len(test_loader)))
    
    # Define loss function (criterion) and optimizer
    criterion = t.nn.CrossEntropyLoss().to(args.device.type)

    # Create the model ( download )
    model = create_model(args)

    for name, module in model.named_modules():
        print(module)
        break
        
    # data Parallel by new model
    if args.device.gpu and not args.dataloader.serialized:
        model = t.nn.DataParallel(model, device_ids=args.device.gpu)

    tbmonitor.writer.add_graph(model, input_to_model=train_loader.dataset[0][0].unsqueeze(0))
    model.to(args.device.type)

    start_epoch = 0

    if args.resume.path:
        model, start_epoch, _ = util.load_checkpoint(
            model, args.resume.path, args.device.type, lean=args.resume.lean)

    # optimizer = t.optim.Adam(model.parameters(), lr=args.optimizer.learning_rate)
    optimizer = t.optim.SGD(model.parameters(),
                            lr=args.optimizer.learning_rate,
                            momentum=args.optimizer.momentum,
                            weight_decay=args.optimizer.weight_decay)
    lr_scheduler = util.lr_scheduler(optimizer,
                                     batch_size=train_loader.batch_size,
                                     num_samples=len(train_loader.sampler),
                                     **args.lr_scheduler)
    logger.info(('Optimizer: %s' % optimizer).replace('\n', '\n' + ' ' * 11))
    logger.info('LR scheduler: %s\n' % lr_scheduler)

    print(f'**********train_loader_sampler : {len(train_loader.sampler)}')

    perf_scoreboard = process.PerformanceScoreboard(args.log.num_best_scores)

    if args.eval:
        process.validate(test_loader, model, criterion, -1, monitors, args)
    else:  # training
        if args.resume.path or args.pre_trained:
            logger.info('>>>>>>>> Epoch -1 (pre-trained model evaluation)')
            top1, top5, _ = process.validate(val_loader, model, criterion,
                                             start_epoch - 1, monitors, args)
            perf_scoreboard.update(top1, top5, start_epoch - 1)
        for epoch in range(start_epoch, args.epochs):
            logger.info('>>>>>>>> Epoch %3d' % epoch)
            t_top1, t_top5, t_loss = process.train(train_loader, model, criterion, optimizer,
                                                   lr_scheduler, epoch, monitors, args)
            v_top1, v_top5, v_loss = process.validate(val_loader, model, criterion, epoch, monitors, args)

            # tbmonitor.writer.add_scalars('Train_vs_Validation/Loss', {'train': t_loss, 'val': v_loss}, epoch)
            # tbmonitor.writer.add_scalars('Train_vs_Validation/Top1', {'train': t_top1, 'val': v_top1}, epoch)
            # tbmonitor.writer.add_scalars('Train_vs_Validation/Top5', {'train': t_top5, 'val': v_top5}, epoch)

            perf_scoreboard.update(v_top1, v_top5, epoch)
            is_best = perf_scoreboard.is_best(epoch)
            util.save_checkpoint(epoch, args.arch, model, {'top1': v_top1, 'top5': v_top5}, is_best, args.name, log_dir)

        logger.info('>>>>>>>> Epoch -1 (final model evaluation)')
        process.validate(test_loader, model, criterion, -1, monitors, args)
    
    
    if args.save_all is True:
        filename = 'raw_model.pth'
        filepath = os.path.join(output_dir, filename)

    # tbmonitor.writer.close()  # close the TensorBoard
    logger.info('Program completed successfully ... exiting ...')
    logger.info('If you have any questions or suggestions, please visit: github.com/zhutmost/lsq-net')




if __name__ == "__main__":
    script_dir = Path.cwd()
    p = argparse.ArgumentParser(description='Learned Step Size Quantization')
    p.add_argument('config_file', metavar='PATH', nargs='+',
                   help='path to a configuration file')
    args = p.parse_args()

    for arg in args.config_file:
        with open(script_dir / str(arg)) as yaml_file:
            cfg = yaml.safe_load(yaml_file)
        config_arg = munch.munchify(cfg)
        # config_arg['epochs'] = 1
        main(config_arg)
    # args = util.get_config(default_file=files)
    # new_args = copy.deepcopy(args)
    
    # to parameter sweep
    '''
    for i in range(5):
        new_args.lr_scheduler['amp_scale'] = args.lr_scheduler['amp_scale'] + i*0.1
        for i in range(5):
            new_args.lr_scheduler['cycle'] = args.lr_scheduler['cycle'] + i*6
            main(new_args)'''

    # main(new_args)
    # train_full_precision(args)


