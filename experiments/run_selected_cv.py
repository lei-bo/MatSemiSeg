from argparse import Namespace

from run_cv import CrossValidation
from segmentation.args import Arguments


class SelectedCV(CrossValidation):
    train_split = 0
    val_split = 1
    test_split = 2
    n_cross_valid = 6
    train_reverse = False

    def __init__(self, args):
        super().__init__(args)
        if not hasattr(args, 'select_method'):
            args.split_info.split_file = f"select_{args.n_select}.csv"
        else:
            args.split_info.split_file = f"{args.select_method}_select_{args.n_select}.csv"

    @classmethod
    def update_args(cls, args, cv_id):
        args_i = Namespace(**vars(args))
        args_i.split_info.train_split_num = cls.train_split
        args_i.split_info.val_split_num = cls.val_split
        args_i.split_info.test_split_num = cls.test_split
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
