from tqdm import tqdm
import torch
import os, logging
from dependency.MKCNet.utils.args import *
import torch.multiprocessing
# from dependency.MKCNet.utils.evaluate import model_validate
# from dependency.MKCNet.dataset.dataset_manager import get_dataloader
try:
    from dependency.MKCNet.model.model_manager import get_model
except Exception as e:
    assert False, e
# from dependency.MKCNet.utils.misc import init_log, check_path
# from dependency.MKCNet.utils.train import *



def pretrain(ckpt):
    args = get_args()
    cfg = setup_cfg(args)
    psi = [cfg.MODEL.META_LENGTH] * cfg.DATASET.NUM_M
    model, metalearner = get_model(cfg, psi)
    model.load_state_dict(torch.load(ckpt))
    return model, cfg
    # return cfg


from torchvision import transforms
def get_transform(cfg=None):

    # means = cfg.DATASET.NORMALIZATION_MEAN
    # std = cfg.DATASET.NORMALIZATION_STD
    means = torch.tensor([0.3771, 0.2320, 0.1395]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor([0.1252, 0.0857, 0.0814]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)


    return means, std

    transfrom_train = []
    transfrom_test = []

    # if dataset in ['DRAC', 'IQAD_CXR', 'IQAD_CT']:
    #     transfrom_train.append(transforms.Grayscale(1))
    #     transfrom_test.append(transforms.Grayscale(1))

    transfrom_train.append(transforms.Resize((256, 256)))
    transfrom_test.append(transforms.Resize((256, 256)))

    transfrom_train.append(transforms.ToTensor())
    transfrom_train.append(transforms.Normalize(means, std))

    transfrom_test.append(transforms.ToTensor())
    transfrom_test.append(transforms.Normalize(means, std))

    train_ts =  transforms.Compose(transfrom_train)
    test_ts = transforms.Compose(transfrom_test)

    return train_ts, test_ts

















if __name__ == "__main__":
    args = get_args()
    cfg = setup_cfg(args)

    log_path = os.path.join('./result', args.output)
    check_path(log_path, args)

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    if not args.random:
        setup_seed(args.seed)
    
    if args.dataset in cfg.VALIDATION_DATASET:
        train_loader, test_loader, val_loader, dataset_size = get_dataloader(cfg)
    else:
        train_loader, test_loader, dataset_size = get_dataloader(cfg)
    
    if args.model not in cfg.MKCNET_MODEL_LIST:
        model = get_model(cfg)

    elif args.model == 'FirstOrder_MKCNet':
        psi = [cfg.MODEL.META_LENGTH] * cfg.DATASET.NUM_M
        model, metalearner, model_compute = get_model(cfg, psi)
        metalearner.cuda()
        metalearner.train()

    elif args.model == 'MKCNet':
        psi = [cfg.MODEL.META_LENGTH] * cfg.DATASET.NUM_M
        model, metalearner = get_model(cfg, psi)
        metalearner.cuda()
        metalearner.train()

    model.cuda()
    model.train()
    
    optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=cfg.OPTIM.LR ,
                    weight_decay=cfg.OPTIM.WEIGHT_DECAY)

    if args.model in cfg.MKCNET_MODEL_LIST:
        optimizer_gen = torch.optim.SGD(
                    metalearner.parameters(),
                    lr=cfg.OPTIM.META_LR,
                    weight_decay=cfg.OPTIM.META_WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.OPTIM.STEP_SIZE, gamma=0.5)

    if args.model in cfg.MKCNET_MODEL_LIST:
        scheduler_gen = torch.optim.lr_scheduler.StepLR(optimizer_gen, step_size=cfg.OPTIM.STEP_SIZE, gamma=0.5)

    criterion = torch.nn.CrossEntropyLoss()

    writer = init_log(args, log_path, train_loader, dataset_size)

    iterator = tqdm(range(cfg.MAX_EPOCHS))
    max_iterations = cfg.MAX_EPOCHS * len(train_loader)

    best_performance = 0.0
    best_performance_test = 0.0

    for i in iterator:
        
        epoch = i + 1
        loss_temp = 0

        if args.model not in cfg.MKCNET_MODEL_LIST:
            loss = train(model, train_loader, criterion, optimizer, writer, epoch, cfg)

        elif args.model == 'MKCNet':
            loss = train_MKCNet(model, metalearner, train_loader, scheduler.get_last_lr()[0], optimizer, optimizer_gen, writer, epoch, cfg)

        elif args.model == 'FirstOrder_MKCNet':
            loss = train_MKCNet_firstorder(model, metalearner, model_compute, train_loader, scheduler.get_last_lr()[0], optimizer, optimizer_gen, writer, epoch, cfg)
        
        logging.info('epoch: {}, loss: {}.'.format(epoch, loss))
        writer.add_scalar('info/lr', scheduler.get_last_lr()[0], epoch)

        scheduler.step()
        if args.model in cfg.MKCNET_MODEL_LIST:
            scheduler_gen.step()

        if epoch % cfg.VAL_STEP == 0:

            if args.dataset in cfg.VALIDATION_DATASET:
                val_auc, val_loss = model_validate(model, val_loader, writer, criterion, epoch, 'val', cfg)
                test_auc, test_loss = model_validate(model, test_loader, writer, criterion, epoch, 'test', cfg)
            else:
                val_auc, val_loss = model_validate(model, test_loader, writer, criterion, epoch, 'test', cfg)

            if val_auc > best_performance:
                best_performance = val_auc
                logging.info("Saving best model...")
                torch.save(model.state_dict(), os.path.join(log_path, 'best_model.pth'))
                if args.model in cfg.MKCNET_MODEL_LIST:
                    torch.save(metalearner.state_dict(), os.path.join(log_path, 'best_model_LG.pth'))
    
        if epoch == cfg.MAX_EPOCHS:
            logging.info("Saving last model...")
            torch.save(model.state_dict(), os.path.join(log_path, 'last_model.pth'))
            if args.model in cfg.MKCNET_MODEL_LIST:
                torch.save(metalearner.state_dict(), os.path.join(log_path, 'last_model_LG.pth'))
    
    model.load_state_dict(torch.load(os.path.join(log_path, 'best_model.pth')))
    if args.dataset in cfg.VALIDATION_DATASET:
        val_auc, val_loss = model_validate(model, val_loader, writer, criterion, cfg.MAX_EPOCHS + cfg.VAL_STEP, 'val', cfg)
    test_auc, test_loss = model_validate(model, test_loader, writer, criterion, cfg.MAX_EPOCHS + cfg.VAL_STEP, 'test', cfg)

    logging.info('Best performance on val: {}'.format(best_performance))
    logging.info('Best performance on test: {}'.format(test_auc))
    os.mknod(os.path.join(log_path, 'done'))
    writer.close()