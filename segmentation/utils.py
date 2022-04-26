import numpy as np
import pandas as pd
import torch
import torch.optim as optim


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, weight=1):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    @property
    def average(self):
        return np.round(self.avg, 5)


class ScoreMeter:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, pred, label):
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                self.confusion_matrix[j][i] += np.logical_and(pred==i, label==j).sum()

    def get_scores(self, verbose=False):
        eps = 1e-8
        cm = self.confusion_matrix
        precision = cm.diagonal() / (cm.sum(axis=0) + eps)
        recall = cm.diagonal() / (cm.sum(axis=1) + eps)
        fraction_error = (cm.sum(axis=0) - cm.sum(axis=1)) / cm.sum()
        iou = cm.diagonal() / (cm.sum(axis=1) + cm.sum(axis=0) - cm.diagonal())
        acc = cm.diagonal().sum() / cm.sum()
        miou = iou.mean()
        score_dict = {
            'accuracy': acc,
            'mIoU': miou,
            'IoUs': iou,
            'precision': precision,
            'recall': recall,
            'fraction_error': fraction_error
        }
        if verbose:
            print('\n'.join(f"{k}: {v:.5f}") for k, v in score_dict.items())
        return score_dict


class Recorder(object):
    def __init__(self, headers):
        self.headers = headers
        self.record = {}
        for header in self.headers:
            self.record[header] = []

    def update(self, vals):
        for header, val in zip(self.headers, vals):
            self.record[header].append(val)

    def save(self, path):
        pd.DataFrame(self.record).to_csv(path, index=False)


class ModelSaver:
    """A helper class to save the model with the best validation miou and save
    the model using early stopping strategy with a given patience."""
    def __init__(self, model_dir, patience, delta=0):
        """
        :param model_dir: the dir to save the model to
        :param patience: the number of epochs to wait before early stopping
        :param delta: minimum change in the monitored quantity to qualify as an
        improvement, defaults to 0
        """
        self.best_miou_path = model_dir.best_miou
        self.early_stop_path = model_dir.early_stop
        self.patience = patience
        self.counter = 0
        self.best_score = np.NINF
        self.early_stop = False
        self.delta = delta
        self.best_miou_epoch, self.early_stop_epoch = 0, 0
        self.early_stop_score = self.best_score

    def save_models(self, score, epoch, args, model, ious):
        if score > self.best_score + self.delta:
            print(f"validation iou improved from {self.best_score:.5f} to {score:.5f}.")
            self.best_score = score
            self.save_checkpoint(self.best_miou_path, epoch, args, model, ious)
            self.best_miou_epoch = epoch
            if not self.early_stop:
                self.save_checkpoint(self.early_stop_path, epoch, args, model, ious)
                self.early_stop_epoch = epoch
                self.early_stop_score = score
                self.counter = 0
        else:
            if not self.early_stop:
                self.counter += 1
                if self.counter >= self.patience:
                    print("early stopping model exceeded patience")
                    self.early_stop = True

    @staticmethod
    def save_checkpoint(path, epoch, args, model, ious):
        torch.save({
            'epoch': epoch,
            'args': args,
            'model_state_dict': model.state_dict(),
            'ious': ious
        }, path)


class LRScheduler:
    LR_SCHEDULER_MAP = {
        'CAWR': optim.lr_scheduler.CosineAnnealingWarmRestarts,
        'MultiStepLR': optim.lr_scheduler.MultiStepLR,
        'CyclicLR': optim.lr_scheduler.CyclicLR,
        'OneCycleLR': optim.lr_scheduler.OneCycleLR
    }
    STEP_EVERY_BATCH = ('CAWR', 'CyclicLR', 'OneCycleLR')

    def __init__(self, lr_scheduler_args, optimizer):
        args = lr_scheduler_args
        self.no_scheduler = False
        if args is None:
            self.no_scheduler = True
            return
        if args.type not in self.LR_SCHEDULER_MAP:
            raise ValueError(f"unsupported lr scheduler: {args.type}")
        else:
            self.lr_scheduler = self.LR_SCHEDULER_MAP[args.type](
                optimizer, **args.params
            )
        self.step_every_batch = args.type in self.STEP_EVERY_BATCH

    def step(self, last_batch=False):
        if self.no_scheduler:
            return
        if self.step_every_batch:
            self.lr_scheduler.step()
        else:
            if last_batch:
                self.lr_scheduler.step()


def get_optimizer(optimizer_args, model):
    args = optimizer_args
    list_params = [{'params': model.encoder.parameters(),
                    'lr': args.encoder_lr,
                    'weight_decay': args.weight_decay},
                   {'params': model.decoder.parameters(),
                    'lr': args.decoder_lr,
                    'weight_decay': args.weight_decay}]
    if args.type == 'Adam':
        optimizer = optim.Adam(list_params)
    elif args.type == 'AdamW':
        optimizer = optim.AdamW(list_params)
    elif args.type == 'SGD':
        optimizer = optim.SGD(list_params)
    else:
        raise ValueError(f"unsupported optimizer: {args.type}")
    return optimizer
