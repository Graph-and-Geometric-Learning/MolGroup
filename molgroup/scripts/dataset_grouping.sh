
python app/train_molgroup.py \
    --device 0 \
    --seeds 0 \
    --dataset_root /home/huangtin/MolGroup/dataset \
    --save_dir /home/huangtin/MolGroup/checkpoints \
    --datasets qm8 ogbg-molbbbp \
    --gate_emb_dim 16 \
    --epochs 10 \
    --feature simple \
    --lr 1e-3 \
    --batch_size 128 \
    --eval_batch_size 1024 \
    --emb_dim 300 \
    --num_layer 5 \
    --drop_ratio 0.5 \
    --eval_step 1 \
    --num_workers 1 \
    --gate_temp 0.1 \
    --gate_mix_alpha 0.1 \
    --use_fp \
    --fp_feat macc \
