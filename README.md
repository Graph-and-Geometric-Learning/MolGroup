# Learning to Group Auxiliary Datasets for Molecule

This is our PyTorch implementation for the paper:

> Tinglin Huang, Ziniu Hu, and Rex Ying (2023). Learning to Group Auxiliary Datasets for Molecule. [Paper in arXiv](https://arxiv.org/abs/2307.04052). In NeurIPS'2023, New Orleans, USA, Dec 10-16, 2023.

## Dataset preparation

```
cd dataset
wget https://raw.githubusercontent.com/snap-stanford/ogb/master/ogb/graphproppred/master.csv
mkdir qm8
cd qm8
wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm8.csv
cd ..
mkdir qm9
cd qm9
wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv
```

## Environment Requirement

The code has been tested running under Python 3.9.16. The required packages are as follows:

* pytorch == 1.13.1+cu117
* torch_geometric == 2.3.1
* ogb == 1.3.6
* rdkit == 2023.03.2
* pandas == 1.3.1
* cython == 3.0.0

Once you finished these installation, please run `pip install -e .`

## Run the code

* Run all combinations of custom dataset pairs (upsample/downsample training instances)
  * `./scripts/all_pairs.sh`
* Pretrain graphormer on pcqm4mv2 dataset
  * `./scripts/pretrain_graphormer.sh`
* Run molgroup example
  * `./scripts/dataset_grouping.sh`
* Run an example of vanilla GIN with dataset combination:
  * `./scripts/example_gin.sh`
* Run an example of pretrained Graphormer with dataset combination:
  * `./scripts/example_graphormer.sh`

The description of the hyperparameters can be found in the utils/utils.py file. The hyperparameter `datasets` is a list of datasets' names (e.g., ogbg-molbbbp ogbg-molfreesolv), inciding the datasets to be used for dataset combination. Specifically, the first dataset in the list will be considered as the target dataset.

## Citation 

If you want to use our codes and datasets in your research, please cite:

```
@inproceedings{huang2023learning,
  author    = {Tinglin Huang and 
              Ziniu Hu and
              Rex Ying},
  title     = {Learning to Group Auxiliary Datasets for Molecule},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2023}
}
```
