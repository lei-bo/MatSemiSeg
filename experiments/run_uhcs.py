import sys
import os
import torch
from argparse import Namespace
import numpy as np

sys.path.append("./")
from segmentation.args import Arguments
from segmentation.train import train
from segmentation.eval import evaluate


class CrossValidation:
    val_splits = [1, 2, 3, 4, 5, 0]
    test_splits = [0, 1, 2, 3, 4, 5]
    n_cross_valid = len(val_splits)

    def __init__(self, args):
        s_info = args.split_info
        s_info.type = 'CSVSplit'
        s_info.test_type = 'CSVSplit'
        s_info.split_file = 'split_6fold.csv'
        s_info.split_col_name = 'split'
        s_info.train_reverse = True
        self.args = args

    @classmethod
    def update_args(cls, args, cv_id):
        args_i = Namespace(**vars(args))
        args_i.split_info.val_split_num = cls.val_splits[cv_id]
        args_i.split_info.test_split_num = cls.test_splits[cv_id]
        args_i.experim_name += f"_CV{cv_id}"
        args_i.checkpoints_dir = f"{args.checkpoints_dir}/CV{cv_id}"
        os.makedirs(args_i.checkpoints_dir, exist_ok=True)
        args_i.model_path = Namespace(
            **{"early_stop": f"{args_i.checkpoints_dir}/early_stop.pth",
               "best_miou": f"{args_i.checkpoints_dir}/best_miou.pth"})
        args_i.record_path = f"{args_i.checkpoints_dir}/train_record.csv"
        return args_i

    def train(self):
        for i in range(self.n_cross_valid):
            args_i = self.update_args(self.args, i)
            Arguments.print_args(args_i)
            train(args_i)

    def evaluate(self, is_test=False):
        best_mious_cv = np.zeros((self.n_cross_valid, self.args.n_classes))
        es_mious_cv = np.zeros((self.n_cross_valid, self.args.n_classes))
        for i in range(self.n_cross_valid):
            args_i = self.update_args(self.args, i)
            if is_test:
                _, best_ious = evaluate(args_i, mode='test', model_type='best_miou')
                _, es_ious = evaluate(args_i, mode='test', model_type='early_stop')
            else:
                best_ious = torch.load(args_i.model_path.best_miou)['ious']['val']
                es_ious = torch.load(args_i.model_path.early_stop)['ious']['val']
            best_mious_cv[i, :] = best_ious
            es_mious_cv[i, :] = es_ious
        title = 'test' if is_test else 'validate'
        print_mean_std(best_mious_cv, f"{title} best_mious")
        print_mean_std(es_mious_cv, f"{title} early_stop")


def print_mean_std(arr, title=None):
    print(title)
    fmt = lambda x: np.round(x * 100, 1)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    for i in range(arr.shape[1]):
        print(f"class {i}: {fmt(mean[i])} +/- {fmt(std[i])}")
    print(f"total: {arr.mean()*100:.1f} +/- {arr.mean(axis=1).std()*100:.1f}")


if __name__ == '__main__':
    arg_parser = Arguments()
    arg_parser.parser.add_argument('--mode', '-m',
                                   choices=['train', 'val', 'test'],
                                   required=True)
    args = arg_parser.parse_args()
    cv = CrossValidation(args)
    if args.mode == 'train':
        cv.train()
    elif args.mode == 'val':
        cv.evaluate(is_test=False)
    elif args.mode == 'test':
        cv.evaluate(is_test=True)
