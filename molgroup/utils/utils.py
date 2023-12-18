import wandb
import torch
import numpy as np
import random
import argparse


def log_func(save_path, record_dict, use_wandb=False):
    assert isinstance(record_dict, dict)
    if use_wandb:
        wandb.log(record_dict)
    else:
        f=open(save_path, 'a')
        save_str = " ".join(list(map(str, record_dict.values())))
        save_str += " \n"
        f.write(save_str)
        f.close()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('--seeds', type=int, nargs="+", default=[0])
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--eval_batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--eval_step', type=int, default=1,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--datasets', type=str, nargs="+", default=["ogbg-mollipo"],
                        help='dataset name (default: ogbg-molhiv)')
    parser.add_argument('--n_train_graphs', type=int, default=5000)
    parser.add_argument('--feature', type=str, default="simple",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    parser.add_argument('--dataset_root', type=str, default="",
                        help='filename to output result (default: )')
    parser.add_argument('--save_dir', type=str, default="",
                        help='filename to output result (default: )')
    parser.add_argument("--balance_sample", action="store_true", default=False)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--save_checkpoint", action="store_true", default=False)

    parser.add_argument('--fp_feat', type=str, default="mgf", help='mgf, macc, fp')

    parser.add_argument("--eval_flag", action="store_true", default=False)
    parser.add_argument('--gate_temp', type=float, default=1.)
    parser.add_argument('--gate_emb_dim', type=int, default=32)
    parser.add_argument('--gate_mix_alpha', type=float, default=0.1)

    # for graphormer
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--attention_dropout_rate', type=float, default=0.1)

    return parser


def build_args():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    return parser.parse_args()