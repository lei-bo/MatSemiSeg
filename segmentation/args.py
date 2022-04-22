from argparse import ArgumentParser, Namespace
import yaml
import os
import torch

class Arguments:
    def __init__(self):
        parser = ArgumentParser()
        parser.add_argument("--root", default='.')
        parser.add_argument("--dataset", default="uhcs")
        parser.add_argument("--config", default="default.yaml")
        parser.add_argument("--gpu_id", type=int, default=0)
        parser.add_argument("--seed", type=int, default=42)
        self.parser = parser

    def parse_args(self):
        args = self.parser.parse_args()
        args.device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

        # load default config and specific config to args for the dataset
        default_config_path = f"{args.root}/segmentation/configs/{args.dataset}/default.yaml"
        default_config = yaml.safe_load(open(f"{default_config_path}", 'r'))
        config_path = f"{args.root}/segmentation/configs/{args.dataset}/{args.config}"
        config = yaml.safe_load(open(f"{config_path}", 'r'))
        args = vars(args)
        args.update(default_config)
        args.update(config)
        args['split_info'] = Namespace(**args['split_info'])
        args = Namespace(**args)

        # compile basic information
        args.dataset_root = f"{args.root}/data/{args.dataset}"
        assert os.path.exists(args.dataset_root), FileNotFoundError(args.dataset_root)
        args.img_dir = f"{args.dataset_root}/{args.img_folder}"
        args.label_dir = f"{args.dataset_root}/{args.label_folder}"

        # TODO: exp name
        args.experim_name = None
        args.checkpoints_dir = f"{args.root}/segmentation/checkpoints/{args.dataset}/{args.experim_name}"
        os.makedirs(args.checkpoints_dir, exist_ok=True)
        args.model_path = Namespace(**{"early_stop": f"{args.checkpoints_dir}/early_stop.pth",
                                       "best_miou": f"{args.checkpoints_dir}/best_miou.pth"})

        return args


if __name__ == '__main__':
    arg_parser = Arguments()
    args = arg_parser.parse_args()
    [print(k,':',v) for k, v in vars(args).items()]