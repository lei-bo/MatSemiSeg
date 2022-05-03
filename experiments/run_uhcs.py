import sys
import os
from argparse import Namespace
import numpy as np
import pickle

sys.path.append("./")
from segmentation.args import Arguments
from segmentation.train import train
from segmentation.eval import evaluate


class CrossValidation:
    val_splits = [1, 2, 3, 4, 5, 0]
    test_splits = [0, 1, 2, 3, 4, 5]
    n_cross_valid = len(val_splits)
    split_file = 'split_cv.csv'
    split_col_name = 'split'
    train_reverse = True

    def __init__(self, args):
        s_info = args.split_info
        s_info.type = 'CSVSplit'
        s_info.test_type = 'CSVSplit'
        s_info.split_file = self.split_file
        s_info.split_col_name = self.split_col_name
        s_info.train_reverse = self.train_reverse
        self.args = args

    @classmethod
    def update_args(cls, args, cv_id):
        args_i = Namespace(**vars(args))
        args_i.split_info.val_split_num = cls.val_splits[cv_id]
        args_i.split_info.test_split_num = cls.test_splits[cv_id]
        args_i.experim_name += f"_CV{cv_id}"
        Arguments.update_checkpoints_dir(args_i, f"{args.checkpoints_dir}/CV{cv_id}")
        return args_i

    def train(self):
        for i in range(self.n_cross_valid):
            args_i = self.update_args(self.args, i)
            Arguments.print_args(args_i)
            train(args_i)

    def evaluate(self, mode='val'):
        mious_cv = np.zeros((self.n_cross_valid, self.args.n_classes))
        for i in range(self.n_cross_valid):
            args_i = self.update_args(self.args, i)
            result_path = args_i.test_result_path if mode == 'test' else args_i.val_result_path
            if os.path.exists(result_path):
                with open(result_path, 'rb') as f:
                    scores = pickle.load(f)
            else:
                scores = evaluate(args_i, mode, save_pred=True)
                with open(result_path, 'wb') as f:
                    pickle.dump(scores, f)
            # print(f"{scores['mIoU']:.5f}")
            mious_cv[i, :] = scores['IoUs']
        print_mean_std(mious_cv, f"{self.args.experim_name} {mode}")

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
    else:
        cv.evaluate(args.mode)
