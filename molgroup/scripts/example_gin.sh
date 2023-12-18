
python app/train.py \
    --device 0 \
    --seeds 0 1 2 3 4 \
    --dataset_root /home/huangtin/MolGroup/dataset \
    --save_dir /home/huangtin/MolGroup/checkpoints \
    --datasets ogbg-molbbbp ogbg-molhiv \
    --gnn gin \
    --epochs 50 \
    --feature simple \
    --n_train_graphs 5000 \
    --lr 1e-3 \
    --batch_size 64 \
    --eval_batch_size 1024 \
    --emb_dim 300 \
    --num_layer 5 \
    --drop_ratio 0.5 \
    --eval_step 1 \
    --num_workers 1 \
    --eval_flag \
