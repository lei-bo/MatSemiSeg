from argparse import Namespace
import pickle, os

from run_cv import CrossValidation
from segmentation.args import Arguments


class SelectedCV(CrossValidation):

    def __init__(self, args):
        super().__init__(args)
        args.split_info.train_split_num = 0
        args.split_info.val_split_num = 1
        args.split_info.test_split_num = 2
        args.split_info.train_reverse = False
        self.n_cross_valid = args.split_info.n_cross_valid

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
        cv.evaluate('val')
        cv.evaluate('test')
    else:
        ious_cv, ferrors_cv = cv.evaluate(args.mode)
        result_path = f"./experiments/results/{args.dataset}_{args.mode}_ious.pkl"
        if not os.path.exists(result_path):
            results = {}
        else:
            with open(f"./experiments/results/{args.dataset}_{args.mode}_ious.pkl", 'rb') as f:
                results = pickle.load(f)
        results[args.experim_name] = {'ferrors': ferrors_cv, 'ious': ious_cv}
        with open(f"./experiments/results/{args.dataset}_{args.mode}_ious.pkl", 'wb') as f:
            pickle.dump(results, f)