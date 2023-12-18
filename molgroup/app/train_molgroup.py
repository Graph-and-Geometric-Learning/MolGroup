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

from molgroup.models import NNDecoder, MetaGIN
from molgroup.utils import get_fingerprint, set_random_seed, build_args
from molgroup.utils.evaluator import OGBEvaluator
from molgroup.utils.dataloader import DataLoaderFP, MixedDataLoader
from molgroup.utils.dataset import CustomPygDataset, QMMolDataset

from ogb.graphproppred import PygGraphPropPredDataset

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def meta_train(epoch, model_list, device, loader, optimizer_list, task_types):
    model, decoder = model_list
    optimizer, gate_optimizer, dec_optimizer = optimizer_list

    model.train()
    decoder.train()

    clf_criterion = torch.nn.BCEWithLogitsLoss()
    reg_criterion = torch.nn.MSELoss()

    n_tasks = len(task_types)
    total_loss_list = [0. for _ in range(n_tasks)]
    epoch_iter = tqdm(loader, ncols=130)
    gates = [0. for _ in range(n_tasks - 1)]
    task_affinitys = [0. for _ in range(n_tasks - 1)]
    structure_affinitys = [0. for _ in range(n_tasks - 1)]
    target_batch = None

    for step, batch_list in enumerate(epoch_iter):
        batch_list = [batch.to(device) if batch is not None else None for batch in batch_list]
        loss_list = [0. for _ in range(n_tasks)]
        
        for dataset_id, batch in enumerate(batch_list):
            if batch is not None:
                if batch.x.shape[0] == 1:
                    pass
                else:
                    if dataset_id == 0:
                        node_rep = model(batch, dataset_id, use_gate=True, return_gate_score=False)
                    else:
                        node_rep, gate = model(batch, dataset_id, use_gate=True, return_gate_score=True, 
                                        target_x=target_batch.fp_feat, target_batch_data=target_batch.batch)

                        gate, task_affinity, structure_affinity = gate
                        task_affinitys[dataset_id - 1] += task_affinity.detach().cpu().item()
                        structure_affinitys[dataset_id - 1] += structure_affinity.detach().cpu().item()
                        gates[dataset_id - 1] += gate.detach().cpu().item()

                    pred = decoder(batch, node_rep, dataset_id)
                    ## ignore nan targets (unlabeled) when computing training loss.
                    is_labeled = batch.y == batch.y
                    criterion = clf_criterion if "classification" in task_types[dataset_id] else reg_criterion
                    loss_list[dataset_id] = criterion(pred.float()[is_labeled], batch.y.float()[is_labeled])

                    if dataset_id == 0:
                        target_batch = batch
                        
        # if one of the loss is tensor, then train the model
        if not any([isinstance(loss, torch.Tensor) for loss in loss_list]):
            continue

        gate_params = [param for name, param in model.named_parameters() if 'gate' in name]
        all_params = [param for name, param in model.named_parameters()] + [param for name, param in decoder.named_parameters()]
        param_names = [name for name, param in model.named_parameters()] + [name for name, param in decoder.named_parameters()]

        if not isinstance(loss_list[0], float) and gate_optimizer is not None:
            meta_loss = 0.
            meta_grads = [None for _ in range(n_tasks - 1)]

            if not any([isinstance(loss, torch.Tensor) for loss in loss_list[1:]]):
                continue

            optimizer.zero_grad()
            gate_optimizer.zero_grad()
            dec_optimizer.zero_grad()

            aux_loss = sum(loss_list[1:]) / len(loss_list[1:])
            aux_loss.backward(retain_graph=True, create_graph=True)

            rep = model(target_batch, dataset_idx=0, return_gate_score=False, update=True)
            pred = decoder(target_batch, rep, 0)
            is_labeled = target_batch.y == target_batch.y
            criterion = clf_criterion if "classification" in task_types[0] else reg_criterion
            # calculate the meta loss on the related gate function
            aux_meta_loss = criterion(pred.float()[is_labeled], target_batch.y.float()[is_labeled])
            meta_loss += aux_meta_loss

            model.zero_grad()
            decoder.zero_grad()
            meta_grads = torch.autograd.grad(meta_loss, gate_params, retain_graph=True, allow_unused=True)

        else:
            meta_loss = None
            meta_grads = None

        optimizer.zero_grad()
        if gate_optimizer is not None:
            gate_optimizer.zero_grad()
        dec_optimizer.zero_grad()
        model.zero_grad()
        decoder.zero_grad()

        loss = sum(loss_list) / len(loss_list)
        # loss.backward()
        all_grads = torch.autograd.grad(loss, all_params, retain_graph=True, allow_unused=True)

        # release the computational graph
        if meta_loss is not None:
            t_loss = sum(loss_list) + meta_loss + aux_loss
        else:
            t_loss = sum(loss_list)

        t_loss.backward()
        optimizer.zero_grad()
        if gate_optimizer is not None:
            gate_optimizer.zero_grad()
        dec_optimizer.zero_grad()
        model.zero_grad()
        decoder.zero_grad()

        # update the gnn parameters with the target task and auxiliary task
        # update the gate parameters with the meta gradient
        gate_i = 0
        for i, (name, param, grad) in enumerate(zip(param_names, all_params, all_grads)):
            if 'gate' not in name:
                param.grad = grad
                # pass
            elif meta_grads is not None:
                if meta_grads[gate_i] is not None:
                    param.grad = meta_grads[gate_i] * 1e3
                gate_i += 1

        if gate_optimizer is not None:
            gate_optimizer.step()
        optimizer.step()
        dec_optimizer.step()

        loss_list = [loss_list[i].cpu().item() if not isinstance(loss_list[i], float) else 0. for i in range(len(loss_list))]
        total_loss_list = [total_loss + loss_list[i] for i, total_loss in enumerate(total_loss_list)]

        loss = 0. if loss is None or isinstance(loss, float) else loss.item()
        epoch_iter.set_description(f"epoch: {epoch}, train_loss: {loss:.4f}")
    
        # torch.cuda.empty_cache()
       
    return [total_loss / (step + 1) for total_loss in total_loss_list], \
            [gate / (step + 1) for gate in gates], \
            [gate / (step + 1) for gate in task_affinitys], \
            [gate / (step + 1) for gate in structure_affinitys]


# train the parameters of each dataset only
def train(epoch, model_list, device, loader, optimizer_list, task_types):
    model, decoder = model_list
    optimizer, gate_optimizer, dec_optimizer = optimizer_list

    model.train()
    decoder.train()

    n_tasks = len(task_types)

    clf_criterion = torch.nn.BCEWithLogitsLoss()
    reg_criterion = torch.nn.MSELoss()

    total_loss_list = [0. for _ in range(n_tasks)]
    epoch_iter = tqdm(loader, ncols=130)
    for step, batch_list in enumerate(epoch_iter):
        batch_list = [batch.to(device) if batch is not None else None for batch in batch_list]
        loss_list = [0. for _ in range(n_tasks)]

        for dataset_id, batch in enumerate(batch_list):
            if batch is not None:
                if batch.x.shape[0] == 1:
                    pass
                else:
                    node_rep = model(batch, dataset_id, return_gate_score=False, use_gate=False)
                    pred = decoder(batch, node_rep, dataset_id)
                    ## ignore nan targets (unlabeled) when computing training loss.
                    is_labeled = batch.y == batch.y
                    criterion = clf_criterion if "classification" in task_types[dataset_id] else reg_criterion
                    loss_list[dataset_id] = criterion(pred.float()[is_labeled], batch.y.float()[is_labeled])

        optimizer.zero_grad()
        gate_optimizer.zero_grad()
        dec_optimizer.zero_grad()
        model.zero_grad()
        decoder.zero_grad()

        loss = sum(loss_list) / len(batch_list)
        if isinstance(loss, torch.Tensor):
            loss.backward()
            optimizer.step()
            dec_optimizer.step()

        loss_list = [loss_list[i].cpu().item() if not isinstance(loss_list[i], float) else 0. for i in range(len(loss_list))]
        total_loss_list = [total_loss + loss_list[i] for i, total_loss in enumerate(total_loss_list)]

        loss = 0. if loss is None or isinstance(loss, float) else loss.item()
        epoch_iter.set_description(f"epoch: {epoch}, train_loss: {loss:.4f}")

    return [total_loss / (step + 1) for total_loss in total_loss_list]


@torch.no_grad()
def test(model_list, device, loader, evaluator, return_gate_scores=False):
    model, decoder = model_list

    model.eval()
    decoder.eval()
    y_true = []
    y_pred = []
    gate_scores_list = [[] for _ in range(5)]

    for step, batch in enumerate(loader):
        if isinstance(batch, list):
            batch = batch[0]
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            if return_gate_scores:
                node_rep, gate_score = model(batch, dataset_idx=0, return_gate_score=return_gate_scores)
                for i in range(len(gate_score)):
                    gate_scores_list[i].append(gate_score[i].detach().cpu())
            else:
                node_rep = model(batch, dataset_idx=0)
            pred = decoder(batch, node_rep, task_idx=0)
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}

    if return_gate_scores:
        gate_scores_list = [torch.cat(gate_scores).tolist() for gate_scores in gate_scores_list]
        return evaluator.eval(input_dict), gate_scores_list
    else: 
        return evaluator.eval(input_dict)


def run(seed, iter_step, args):
    set_random_seed(seed)

    save_path = args.save_path
    device = args.device

    # ct: custom dataset
    s = time()
    dataset_list = []
    for dataset in args.datasets:
        if dataset.startswith("ct-"):
            dataset_list.append(CustomPygDataset(name=dataset[3:], root=args.dataset_root, n_train_graphs=args.n_train_graphs))
        elif dataset == "qm8" or dataset == "qm9":
            dataset_list.append(QMMolDataset(name=dataset, root=args.dataset_root))
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
    train_fp_list = []

    for i, (dataset_name, dataset) in enumerate(zip(datasets, dataset_list)):
        if args.feature == 'full':
            pass
        elif args.feature == 'simple':
            # only retain the top two node/edge features
            dataset.data.x = dataset.data.x[:, :2]
            dataset.data.edge_attr = dataset.data.edge_attr[:, :2]

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

        # extract the fingerprint features
        fp_feats = get_fingerprint(dataset_name, args.dataset_root, fp_type=args.fp_feat)
        fp_feats = fp_feats.float()
        train_fp_list.append(fp_feats[train_idx])

        num_tasks_list.append(dataset.num_tasks)
        train_dataset_list.append(dataset[train_idx])

        valid_loader_list.append(DataLoaderFP(dataset[valid_idx], fp_feats[valid_idx] if args.use_fp else None, 
                                dataset_id=0, batch_size=args.eval_batch_size, shuffle=False, num_workers = args.num_workers))
        test_loader_list.append(DataLoaderFP(dataset[test_idx], fp_feats[test_idx] if args.use_fp else None, 
                                dataset_id=0, batch_size=args.eval_batch_size, shuffle=False, num_workers = args.num_workers))

    train_loader = MixedDataLoader(train_dataset_list, train_fp_list if args.use_fp else None, 
                            balance=args.balance, iter_all_graphs=args.iter_all_graphs, 
                            batch_size=args.batch_size, 
                            shuffle=True, num_workers = args.num_workers)

    if args.fp_feat == "mgf": gate_input_dim = 2048
    elif args.fp_feat == "macc": gate_input_dim = 167
    else: gate_input_dim = 2048

    model = MetaGIN(num_layer = args.num_layer, n_datasets = len(dataset_list), 
                    emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, 
                    gate_input_dim=gate_input_dim, gate_hidden_dim=args.gate_emb_dim, 
                    gate_temp=args.gate_temp, gate_mix_alpha=args.gate_mix_alpha).to(device)
    
    decoder = NNDecoder(num_tasks_list=num_tasks_list, emb_dim=args.emb_dim).to(device)
    model_list = [model, decoder]

    optimizer = optim.Adam([param for name, param in model.named_parameters() if 'gate' not in name], lr=args.lr)
    gate_optimizer = optim.Adam([param for name, param in model.named_parameters() if 'gate' in name], lr=args.lr)
    dec_optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    optimizer_list = [optimizer, gate_optimizer, dec_optimizer]

    for param in model.parameters():
        param.lr = args.lr

    train_curve, valid_curve, test_curve = [], [], []

    if args.save_checkpoint and not os.path.exists(save_path):
        os.makedirs(save_path)

    if "meta" in args.gnn:
        gates = model.gating_score().detach().cpu()
        print(gates)  # list: [n_layers, n_tasks]

    for epoch in range(1, args.epochs + 1):
        train_gates = train_task_gates = train_structure_gates = 0.
        if epoch - 1 < args.pretrain_epochs:
            train_perf = train(epoch, model_list, device, train_loader, optimizer_list, task_types)
        else: 
            train_perf, train_gates, train_task_gates, train_structure_gates = meta_train(epoch, model_list, device, train_loader, optimizer_list, task_types)

        train_curve.append(train_perf[0])
        gates = model.gating_score().detach().cpu()
        print(gates)  # list: [n_layers, n_tasks]

        if epoch % args.eval_step == 0:

            ### only evaluate on the first dataset
            valid_perf = test(model_list, device, valid_loader_list[0], evaluator_list[0])
            test_perf = test(model_list, device, test_loader_list[0], evaluator_list[0])

            print({f'Train_{seed}': train_perf, f'Validation_{seed}': valid_perf, f'Test_{seed}': test_perf})
            print({f'Gates_{seed}': train_gates, f'Task_gates_{seed}': train_task_gates, f'Structure_gates_{seed}': train_structure_gates})

            if args.use_wandb:
                wandb.log({f'Train_{seed}': train_perf[0], f'Validation_{seed}': valid_perf, f'Test_{seed}': test_perf})

                for i, gate in enumerate(gates):
                    for j, g in enumerate(gate):
                        wandb.log({f'Gate_layer{i}_{args.datasets[j]}_{iter_step}_{seed}': g})

                if isinstance(train_gates, list):
                    for i, gate in enumerate(train_gates):
                        wandb.log({f'Train_gate_{args.datasets[i+1]}_{iter_step}_{seed}': gate})
                
                if isinstance(train_task_gates, list):
                    for i, gate in enumerate(train_task_gates):
                        wandb.log({f'Train_task_gate_{args.datasets[i+1]}_{iter_step}_{seed}': gate})

                if isinstance(train_structure_gates, list):
                    for i, gate in enumerate(train_structure_gates):
                        wandb.log({f'Train_structure_gate_{args.datasets[i+1]}_{iter_step}_{seed}': gate})

            valid_curve.append(valid_perf[eval_metric_list[0]])
            test_curve.append(test_perf[eval_metric_list[0]])

        else:
            print({f'Train_{seed}': train_perf, f'Gates_{seed}': train_gates, f'Task_gates_{seed}': train_task_gates, f'Structure_gates_{seed}': train_structure_gates})

    if 'classification' in task_types[0]:
        best_val_epoch = np.argmax(np.array(valid_curve))
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))

    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

    if args.use_wandb:
        wandb.log({f'Best validation score, seed {seed}': valid_curve[best_val_epoch], f'Best test score, seed {seed}': test_curve[best_val_epoch]})

    return [0.] + train_gates


def main(args):

    args.balance = True
    args.iter_all_graphs = False
    args.iter_steps = 3
    args.pretrain_epochs = 1

    datasets = [data.split('mol')[1] if "mol" in data else data for data in args.datasets]
    if len(datasets) == 15:
        datasets = datasets[0] + "_all"
    checkpoint_folder = "molgroup" \
                        + "-" + str(datasets) \
                        + "-" + str(args.lr) \
                        + "-" + str(args.feature) \
                        + "-" + str(args.epochs) \
                        + "-" + str(args.drop_ratio) \
                        + "-" + str(args.batch_size) \
                        + "-" + str(args.fp_feat) \
                        + "-" + str(args.gate_temp) \
                        + "-" + str(args.gate_mix_alpha) \
                        + "-" + str(args.pretrain_epochs) \

    args.save_path = os.path.join(args.save_dir, checkpoint_folder)

    """record"""
    if args.use_wandb:
        exp_name = checkpoint_folder 
        wandb.init(project="dataset_grouping", name=exp_name)
        wandb.config.update(args)

    logging.info(f"Save path: {args.save_path}")
    print(args)

    for seed in args.seeds:
        for step in range(args.iter_steps):
            logging.info(f"Iter step: {step}")

            gate_scores = run(seed, step, args)

            gate_save_path = "molgroup" \
                    + "-" + str(seed) \
                    + "-" + str(args.epochs) \
                    + "-" + str(args.batch_size) \
                    + "-" + str(args.gate_temp) \
                    + "-" + str(args.gate_mix_alpha) \
                    + "-" + str(args.use_fp) \
                    + "-" + str(args.fp_feat) \
                    + "-" + str(args.pretrain_epochs) \

            if not os.path.exists(f"results"):
                os.makedirs(f"results")
            f = open(f"results/meta_gates_{args.gnn}_{gate_save_path}.txt", 'a')
            f.write(f"{args.datasets}, {seed}\n{gate_scores}\n")
            f.close()

            gate_scores = np.array(gate_scores)
            # get the index of gate with the score not higher than 0.4
            idx = np.where(gate_scores <= 0.4)[0]
            if len(idx) == len(args.datasets):
                # drop the last dataset
                last_dataset_idx = np.argmax(gate_scores)
                idx = np.concatenate([idx[:last_dataset_idx], idx[last_dataset_idx+1:]])
            args.datasets = [args.datasets[i] for i in idx]
            print(f"Remaining datasets: {args.datasets}")

            if len(args.datasets) == 1:
                break

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    args = build_args()
    logging.info("Args loading.")
    main(args)
