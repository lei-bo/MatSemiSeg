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
    def __init__(self, args):
        self.val_splits = [1, 2, 3, 4, 5, 0]
        self.test_splits = [0, 1, 2, 3, 4, 5]
        self.n_cross_valid = len(self.val_splits)
        s_info = args.split_info
        s_info.type = 'CSVSplit'
        s_info.test_type = 'CSVSplit'
        s_info.split_file = 'split_6fold.csv'
        s_info.split_col_name = 'split'
        s_info.train_reverse = True
        self.args = args

    def update_args(self, cv_id):
        args_i = Namespace(**vars(self.args))
        args_i.split_info.val_split_num = self.val_splits[cv_id]
        args_i.split_info.test_split_num = self.test_splits[cv_id]
        args_i.experim_name += f"_CV{cv_id}"
        args_i.checkpoints_dir = f"{self.args.checkpoints_dir}/CV{cv_id}"
        os.makedirs(args_i.checkpoints_dir, exist_ok=True)
        args_i.model_path = Namespace(
            **{"early_stop": f"{args_i.checkpoints_dir}/early_stop.pth",
               "best_miou": f"{args_i.checkpoints_dir}/best_miou.pth"})
        args_i.record_path = f"{args.checkpoints_dir}/train_record.csv"
        return args_i

    def train(self):
        for i in range(self.n_cross_valid):
            args_i = self.update_args(i)
            Arguments.print_args(args_i)
            train(args_i)

    def validate(self, model_type):
        mious_cv = np.zeros((self.n_cross_valid, self.args.n_classes))
        for i in range(self.n_cross_valid):
            args_i = self.update_args(i)
            if model_type == 'best_miou':
                model_path = args_i.model_path.best_miou
            elif model_type == 'early_stop':
                model_path = args_i.model_path.early_stop
            else:
                raise ValueError
            val_ious = torch.load(model_path)['ious']['val']
            mious_cv[i, :] = val_ious
        print_mean_std(mious_cv)

    def test(self, model_type):
        mious_cv = np.zeros((self.n_cross_valid, self.args.n_classes))
        for i in range(self.n_cross_valid):
            args_i = self.update_args(i)
            test_miou, test_ious = evaluate(args_i, mode='test', model_type=model_type)
            mious_cv[i, :] = test_ious
        print_mean_std(mious_cv)


def print_mean_std(arr):
    print("mean:", arr.mean(axis=0))
    print("std:", arr.std(axis=0))


if __name__ == '__main__':
    arg_parser = Arguments()
    arg_parser.parser.add_argument('--mode', '-m',
                                   choices=['train', 'val', 'test'],
                                   required=True)
    arg_parser.parser.add_argument('--model_type', default='best_miou',
                                   choices=['best_miou', 'early_stop'])
    args = arg_parser.parse_args()
    cv = CrossValidation(args)
    if args.mode == 'train':
        cv.train()
    elif args.mode == 'val':
        cv.validate(args.model_type)
    elif args.mode == 'test':
        cv.test(args.model_type)
