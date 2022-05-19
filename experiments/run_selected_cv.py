from argparse import Namespace

from run_cv import CrossValidation
from segmentation.args import Arguments


class SelectedCV(CrossValidation):

    def __init__(self, args):
        super().__init__(args)
        args.split_info.train_split_num = 0
        args.split_info.val_split_num = 1
        args.split_info.test_split_num = 2
        args.split_info.train_reverse = False

    @classmethod
    def update_args(cls, args, cv_id):
        args_i = Namespace(**vars(args))
        args_i.split_info.split_col_name = f"CV{cv_id}"
        args_i.experim_name += f"_CV{cv_id}"
        Arguments.update_checkpoints_dir(args_i, f"{args.checkpoints_dir}/CV{cv_id}")
        return args_i


if __name__ == '__main__':
    arg_parser = Arguments()
    arg_parser.parser.add_argument('--mode', '-m',
                                   choices=['train', 'val', 'test'],
                                   required=True)
    args = arg_parser.parse_args()
    cv = SelectedCV(args)
    if args.mode == 'train':
        cv.train()
    else:
        cv.evaluate(args.mode)
