import torch
import torch.nn as nn
from tqdm import tqdm

from .args import Arguments
from .UNet import UNetVgg16
from .datasets import get_dataloaders
from .utils import AverageMeter, ScoreMeter


@torch.no_grad()
def eval_epoch(model, dataloader, n_classes, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    score_meter = ScoreMeter(n_classes)
    for inputs, labels, _ in tqdm(dataloader):
        inputs, labels = inputs.to(device), labels.long().to(device)
        # forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        preds = outputs.detach().cpu().numpy().argmax(axis=1)
        # measure
        loss_meter.update(loss.item(), inputs.size(0))
        score_meter.update(preds, labels.cpu().numpy())

    scores = score_meter.get_scores()
    miou, ious, acc = scores['mIoU'], scores['IoUs'], scores['accuracy']
    return loss_meter.avg, acc, miou, ious


def evaluate(args, mode, model_type):
    _, val_loader, test_loader = get_dataloaders(args)
    if mode == 'val':
        dataloader = val_loader
    elif mode == 'test':
        dataloader = test_loader
    else:
        raise ValueError(f"{mode} not supported. Choose from 'val' or 'test'")
    model = UNetVgg16(n_classes=args.n_classes).to(args.device)
    if model_type == 'best_miou':
        model_path = args.model_path.best_miou
    elif model_type == 'early_stop':
        model_path = args.model_path.early_stop
    else:
        raise ValueError(f"{model_type} not supported. Choose from 'best_miou' or 'early_stop'")
    model.load_state_dict(torch.load(model_path)['model_state_dict'], strict=False)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_index).to(args.device)
    eval_loss, eval_acc, eval_miou, eval_ious = eval_epoch(
        model=model,
        dataloader=dataloader,
        n_classes=args.n_classes,
        criterion=criterion,
        device=args.device
    )
    print(f"{mode} | mIoU: {eval_miou:.3f} | accuracy: {eval_acc:.3f} | loss: {eval_loss:.3f}")
    return eval_miou, eval_ious


if __name__ == '__main__':
    arg_parser = Arguments()
    arg_parser.parser.add_argument('--mode', '-m', default='val',
                                   choices=['val', 'test'])
    arg_parser.parser.add_argument('--model_type', default='best_miou',
                                   choices=['best_miou', 'early_stop'])
    args = arg_parser.parse_args()
    evaluate(args, args.mode, args.model_type)
