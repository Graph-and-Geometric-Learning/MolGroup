import os
import yaml
import wandb
import logging
import numpy as np
from tqdm import tqdm
from time import time
from copy import deepcopy

import torch
import torch.optim as optim

from molgroup.models import GIN, Graphormer, NNDecoder
from molgroup.utils import set_random_seed, build_args
from molgroup.utils.evaluator import OGBEvaluator
from molgroup.utils.dataloader import DataLoaderFP, MixedDataLoader, GraphormerMixedDataLoader
from molgroup.utils.dataset import CustomPygDataset, QMMolDataset

from ogb.graphproppred import PygGraphPropPredDataset
from ogb.lsc.pcqm4mv2_pyg import PygPCQM4Mv2Dataset

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


# for loading pretrained graphormer
def load_configs(args):
    config = yaml.load(open("./pretrain_configs/model_configs.yaml", "r"), Loader=yaml.FullLoader)
    param_dict = {}
    cfg = config["graphormer"]
    for item in cfg:
        item = tuple(item.items())[0]
        param_dict[item[0]] = item[1]

    checkpoint_path = f"{args.save_dir}/{str(param_dict['gnn'])}" \
                    + "-" + str([param_dict["datasets"]]) \
                    + "-" + str(param_dict["lr"]) \
                    + "-" + str(param_dict["feature"]) \
                    + "-" + str(param_dict["num_layer"]) \
                    + "-" + str(param_dict["emb_dim"]) \
                    + "-" + str(param_dict['drop_ratio']) \
                    + "-" + str(param_dict['num_heads']) \
                    + "-" + str(param_dict['attention_dropout_rate']) \
                    + "-" + str(param_dict['batch_size']) \
                    + "-" + str(param_dict['balance']) \
                    + "-" + str(param_dict['iter_all_graphs']) \
                    + "/" + str(param_dict['seed']) + "_" + str(param_dict['epoch']) + ".pt" \

    return checkpoint_path


def train(epoch, model_list, device, loader, optimizer_list, task_types,):
    model, decoder = model_list
    optimizer, dec_optimizer = optimizer_list

    model.train()
    decoder.train()

    clf_criterion = torch.nn.BCEWithLogitsLoss()
    reg_criterion = torch.nn.MSELoss()

    total_loss_list = [0. for _ in range(len(task_types))]
    epoch_iter = tqdm(loader, ncols=130)
    for step, batch_list in enumerate(epoch_iter):
        batch_list = [batch.to(device) if batch is not None else None for batch in batch_list]
        loss_list = [0. for _ in range(len(task_types))]
        
        optimizer.zero_grad()
        dec_optimizer.zero_grad()

        for dataset_id, batch in enumerate(batch_list):
            if batch is not None:
                if batch.x.shape[0] == 1:
                    pass
                else:
                    node_rep = model(batch, dataset_id)
                    pred = decoder(batch, node_rep, dataset_id)
                    ## ignore nan targets (unlabeled) when computing training loss.
                    is_labeled = batch.y == batch.y
                    criterion = clf_criterion if "classification" in task_types[dataset_id] else reg_criterion
                    loss_list[dataset_id] = criterion(pred.float()[is_labeled], batch.y.float()[is_labeled])

        loss = sum(loss_list) / len(batch_list)
        if isinstance(loss, torch.Tensor):
            loss.backward()
            optimizer.step()
            dec_optimizer.step()

        loss_list = [loss_list[i].cpu().item() if not isinstance(loss_list[i], float) else 0. for i in range(len(loss_list))]
        total_loss_list = [total_loss + loss_list[i] for i, total_loss in enumerate(total_loss_list)]
        
        loss = 0. if loss is None or isinstance(loss, float) else loss.item()
        epoch_iter.set_description(f"epoch: {epoch}, train_loss: {loss:.4f}")

        if step == 10:
            break
    
    return [total_loss / (step + 1) for total_loss in total_loss_list]


@torch.no_grad()
def test(model_list, device, loader, evaluator):
    model, decoder = model_list

    model.eval()
    decoder.eval()
    y_true, y_pred = [], []

    for step, batch in enumerate(loader):
        if isinstance(batch, list):
            batch = batch[0].to(device)   # for graphormer
        else:
            batch = batch.to(device)  # for gin

        if batch.x.shape[0] == 1:
            pass
        else:
            node_rep = model(batch, 0)
            pred = decoder(batch, node_rep, 0)
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return evaluator.eval(input_dict)


def run(seed, args):
    set_random_seed(seed)

    save_path = args.save_path
    device = args.device

    s = time()
    dataset_list = []
    for dataset in args.datasets:
        if dataset.startswith("ct-"):
            dataset_list.append(CustomPygDataset(name=dataset[3:], root=args.dataset_root, n_train_graphs=args.n_train_graphs))
        elif dataset == "qm8" or dataset == "qm9":
            dataset_list.append(QMMolDataset(name=dataset, root=args.dataset_root))
        elif dataset == "pcqm4mv2":
            dataset_list.append(PygPCQM4Mv2Dataset(root=args.dataset_root))
            dataset_list[-1].task_type = "regression"
            dataset_list[-1].eval_metric = "mae"
            dataset_list[-1].num_tasks = 1
        else:
            dataset_list.append(PygGraphPropPredDataset(name=dataset, root=args.dataset_root))

    logging.info(f"Loaded {len(dataset_list)} datasets in {time() - s:.2f} seconds.")

    datasets = [dataset.replace("ct-", "") for dataset in args.datasets]

    # automatic evaluator. takes dataset name as input
    evaluator_list = [OGBEvaluator(dataset, args.dataset_root) for dataset in datasets]
    task_types = [dataset.task_type for dataset in dataset_list]
    eval_metric_list = [dataset.eval_metric for dataset in dataset_list]

    num_tasks_list = []
    train_dataset_list = []
    valid_loader_list, test_loader_list = [], []

    for i, (dataset_name, dataset) in enumerate(zip(datasets, dataset_list)):
        if args.feature == 'full':
            pass
        elif args.feature == 'simple':
            dataset.data.x = dataset.data.x[:, :2]
            dataset.data.edge_attr = dataset.data.edge_attr[:, :2]

        split_idx = dataset.get_idx_split()
        train_idx = split_idx["train"]
        if dataset_name != "pcqm4mv2":
            valid_idx = split_idx["valid"]
            test_idx = split_idx["test"]
        else:
            # for accerlerating evaluation during pretraining
            valid_idx = split_idx["valid"][:len(split_idx["valid"]) // 2]
            test_idx = split_idx["valid"][len(split_idx["valid"]) // 2:]

        num_tasks_list.append(dataset.num_tasks)
        train_dataset_list.append(dataset[train_idx])

        if args.gnn == "gin":
            valid_loader_list.append(DataLoaderFP(dataset[valid_idx], 
                                    dataset_id=0, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers))
            test_loader_list.append(DataLoaderFP(dataset[test_idx], 
                                    dataset_id=0, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers))
        else:
            valid_loader_list.append(GraphormerMixedDataLoader([dataset[valid_idx]], 
                                    balance=False, iter_all_graphs=True,
                                    batch_size=args.eval_batch_size, 
                                    shuffle=False, num_workers = args.num_workers))
            test_loader_list.append(GraphormerMixedDataLoader([dataset[test_idx]], 
                                    balance=False, iter_all_graphs=True,
                                    batch_size=args.eval_batch_size, 
                                    shuffle=False, num_workers = args.num_workers))

    if args.gnn == "gin":
        # trained with dataset combination
        train_loader = MixedDataLoader(train_dataset_list, 
                                balance = args.balance, iter_all_graphs = args.iter_all_graphs,
                                batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        model = GIN(num_layer=args.num_layer, emb_dim=args.emb_dim, 
                    drop_ratio=args.drop_ratio).to(device)
    else:
        # trained with dataset combination
        train_loader = GraphormerMixedDataLoader(
                                train_dataset_list, 
                                balance=args.balance, iter_all_graphs=args.iter_all_graphs,
                                batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        model = Graphormer(n_atom_feats=9 if args.feature == "full" else 2,
                            n_bond_feats=3 if args.feature == "full" else 2,
                            n_layers=args.num_layer, num_heads=args.num_heads, hidden_dim=args.emb_dim,
                            dropout_rate=args.drop_ratio, intput_dropout_rate=args.drop_ratio, 
                            ffn_dim=args.emb_dim, edge_type="multi_hop", multi_hop_max_dist=5, 
                            attention_dropout_rate=args.attention_dropout_rate).to(device)

    if args.load_checkpoint and args.gnn == "graphormer":
        checkpoint_path = load_configs(args)
        checkpoint = torch.load(checkpoint_path)["model"]
        # model.load_state_dict(checkpoint)
        model.load_pretrained_parameters(checkpoint)
        logging.info("Checkpoint loaded from {}".format(checkpoint_path))
    else:
        logging.info("Training from scratch")

    decoder = NNDecoder(num_tasks_list = num_tasks_list,  emb_dim = args.emb_dim).to(device)
    model_list = [model, decoder]

    filter_fn = filter(lambda p : p.requires_grad, model.parameters())
    optimizer = optim.Adam(filter_fn, lr=args.lr)
    dec_optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    optimizer_list = [optimizer, dec_optimizer]

    train_curve, valid_curve, test_curve = [], [], []

    if args.save_checkpoint and not os.path.exists(save_path):
        os.makedirs(save_path)

    for epoch in range(1, args.epochs + 1):
        train_perf = train(epoch, model_list, device, train_loader, optimizer_list, task_types)
        train_curve.append(train_perf[0])

        if epoch % args.eval_step == 0 and args.eval_flag:
            # only evaluate on the first (target) dataset
            valid_perf = test(model_list, device, valid_loader_list[0], evaluator_list[0])
            test_perf = test(model_list, device, test_loader_list[0], evaluator_list[0])
            
            print({f'Train_{seed}': train_perf, f'Validation_{seed}': valid_perf, f'Test_{seed}': test_perf})

            if args.use_wandb:
                wandb.log({f'Train_{seed}': train_perf[0], f'Validation_{seed}': valid_perf, f'Test_{seed}': test_perf})
            valid_curve.append(valid_perf[eval_metric_list[0]])
            test_curve.append(test_perf[eval_metric_list[0]])
        else:
            print({f'Train_{seed}': train_perf})

        if epoch % 10 == 0 and args.save_checkpoint:
            torch.save({"model": model.cpu().state_dict(), 
                        "decoder": decoder.cpu().state_dict()}, 
                        f"{save_path}/{seed}_{epoch}.pt")
            model.to(device)
            decoder.to(device)

    if 'classification' in task_types[0]:
        best_val_epoch = np.argmax(np.array(valid_curve))
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))

    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))
    return valid_curve[best_val_epoch], test_curve[best_val_epoch]


def main(args):

    datasets = [data.split('mol')[1] if "mol" in data else data for data in args.datasets]
    if len(datasets) > 1 and not args.datasets[0].startswith('ct-'):
        args.balance = True  # sample the instances from each dataset with the same probability
        args.iter_all_graphs = False
    else:
        args.balance = False
        args.iter_all_graphs = True  # iterate over all graphs in all dataset

    if args.gnn == "gin":
        checkpoint_folder = str(args.gnn) \
                            + "-" + str(datasets) \
                            + "-" + str(args.lr) \
                            + "-" + str(args.feature) \
                            + "-" + str(args.num_layer) \
                            + "-" + str(args.emb_dim) \
                            + "-" + str(args.drop_ratio) \
                            + "-" + str(args.batch_size) \
                            + "-" + str(args.balance) \
                            + "-" + str(args.iter_all_graphs) \
    
    else:
        checkpoint_folder = str(args.gnn) \
                            + "-" + str(datasets) \
                            + "-" + str(args.lr) \
                            + "-" + str(args.feature) \
                            + "-" + str(args.num_layer) \
                            + "-" + str(args.emb_dim) \
                            + "-" + str(args.drop_ratio) \
                            + "-" + str(args.num_heads) \
                            + "-" + str(args.attention_dropout_rate) \
                            + "-" + str(args.batch_size) \
                            + "-" + str(args.balance) \
                            + "-" + str(args.iter_all_graphs) \

    args.save_path = os.path.join(args.save_dir, checkpoint_folder)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.dataset_root):
        os.makedirs(args.dataset_root)

    """record"""
    if args.use_wandb:
        exp_name = checkpoint_folder 
        wandb.init(project="dataset_grouping", name=exp_name)
        wandb.config.update(args)

    logging.info(f"Save path: {args.save_path}")
    print(args)

    val_metrics, test_metrics = [], []
    for seed in args.seeds:
        val_metric, test_metric = run(seed, args)
        val_metrics.append(val_metric)
        test_metrics.append(test_metric)

    print(f"Validation: {np.mean(val_metrics):.4f}(±{np.std(val_metrics):.4f})")
    print(f"Test: {np.mean(test_metrics):.4f}(±{np.std(test_metrics):.4f})")

    if args.use_wandb:
        wandb.finish()

    if not os.path.exists(f"results/"):
        os.mkdir(f"results/")
    f = open(f"results/{args.gnn}.txt", 'a')
    f.write(f"{args.datasets}, {args.gnn}, {np.mean(val_metrics):.4f}(±{np.std(val_metrics):.4f}), {np.mean(test_metrics):.4f}(±{np.std(test_metrics):.4f})\n")
    f.close()


if __name__ == "__main__":
    args = build_args()
    logging.info("Args loading.")

    all_datasets = ["ogbg-moltox21", "ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molsider", "ogbg-moltoxcast", 
                       "ogbg-molesol", "ogbg-molfreesolv", "ogbg-mollipo", "qm8", "qm9", "ogbg-molhiv", "ogbg-molpcba", 
                       "ogbg-molmuv", "ogbg-molchembl"]
    target_datasets = ["ogbg-moltox21", "ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molsider", "ogbg-moltoxcast", 
                       "ogbg-molesol", "ogbg-molfreesolv", "ogbg-mollipo", "qm8", "qm9"]

    if args.datasets == ["all"]:
        for dataset in target_datasets:
            args.datasets = [dataset]
            main(args)

    elif args.datasets == ["custom_all"]:
        for dataset in target_datasets:
            args.datasets = ["ct-" + dataset if "ogbg" in dataset else dataset]
            main(args)

    elif args.datasets == ["custom_pairs"]:
        # for the preliminary experiments in the paper
        # The dataset name, which begins with "ct-" (such as ct-ogbg-molfressolv), 
        # indicates that the training set will be upsampled/downsampled to accommodate n_train_graphs (5000 by default) instances.
        for i in range(0, len(target_datasets)):
            for j in range(0, len(all_datasets)):
                if target_datasets[i] != all_datasets[j]:
                    args.datasets = ["ct-" + target_datasets[i] if "ogbg" in target_datasets[i] else target_datasets[i], 
                                     "ct-" + all_datasets[j] if "ogbg" in all_datasets[j] else all_datasets[j]]
                    main(args)

    else:
        main(args)
