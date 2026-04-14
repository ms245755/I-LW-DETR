model_name='lwdetr_tiny_coco'
coco_path=$1
checkpoint=$2

python -u -m torch.distributed.launch \
    --nproc_per_node=4 \
    --use_env \
    main.py \
    --lr 1e-4 \
    --lr_encoder 1.5e-4 \
    --batch_size 1 \
    --weight_decay 1e-4 \
    --epochs 2 \
    --lr_drop 60 \
    --lr_vit_layer_decay 0.8 \
    --lr_component_decay 0.7 \
    --encoder vit_tiny \
    --vit_encoder_num_layers 6 \
    --window_block_indexes 0 2 4 \
    --out_feature_indexes 1 3 5 \
    --dec_layers 3 \
    --group_detr 13 \
    --two_stage \
    --projector_scale P4 \
    --hidden_dim 256 \
    --sa_nheads 8 \
    --ca_nheads 16 \
    --dec_n_points 2 \
    --bbox_reparam \
    --lite_refpoint_refine \
    --num_queries 100 \
    --ia_bce_loss \
    --cls_loss_coef 1 \
    --num_select 100 \
    --dataset_file coco \
    --coco_path $coco_path \
    --square_resize_div_64 \
    --pretrained_encoder pretrain_weights/caev2_tiny_300e_objects365.pth \
    --pretrain_weights pretrain_weights/LWDETR_tiny_30e_objects365.pth \
    --resume $checkpoint \
    --qat \
    --pretrain_keys_modify_to_load transformer.enc_out_class_embed.0.linear.weight transformer.enc_out_class_embed.1.linear.weight transformer.enc_out_class_embed.2.linear.weight transformer.enc_out_class_embed.3.linear.weight transformer.enc_out_class_embed.4.linear.weight transformer.enc_out_class_embed.5.linear.weight transformer.enc_out_class_embed.6.linear.weight transformer.enc_out_class_embed.7.linear.weight transformer.enc_out_class_embed.8.linear.weight transformer.enc_out_class_embed.9.linear.weight transformer.enc_out_class_embed.10.linear.weight transformer.enc_out_class_embed.11.linear.weight transformer.enc_out_class_embed.12.linear.weight transformer.enc_out_class_embed.0.linear.bias transformer.enc_out_class_embed.1.linear.bias transformer.enc_out_class_embed.2.linear.bias transformer.enc_out_class_embed.3.linear.bias transformer.enc_out_class_embed.4.linear.bias transformer.enc_out_class_embed.5.linear.bias transformer.enc_out_class_embed.6.linear.bias transformer.enc_out_class_embed.7.linear.bias transformer.enc_out_class_embed.8.linear.bias transformer.enc_out_class_embed.9.linear.bias transformer.enc_out_class_embed.10.linear.bias transformer.enc_out_class_embed.11.linear.bias transformer.enc_out_class_embed.12.linear.bias class_embed.linear.weight class_embed.linear.bias \
    --output_dir output/$model_name
