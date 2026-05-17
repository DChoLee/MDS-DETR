coco_path=../../data/coco
num_gpus=8

EXP_DIR=exps/train_mds_swinl_900_12e
# no scales.
mkdir -p $EXP_DIR
# python main.py \
GPUS_PER_NODE=$num_gpus ./tools/run_dist_launch.sh $num_gpus python main.py \
   --backbone swin_large \
   --with_box_refine \
   --two_stage \
   --epochs 12 \
   --lr_drop 11 \
   --batch_size 2 \
   --dim_feedforward 2048 \
   --dropout 0.0 \
   --learn_token_bias minus \
   --coco_path=$coco_path \
   --num_queries 900 \
   --use_rel_enc rel_only \
   --use_pos_mlp \
   --dec_layers 5 \
   --separate_mode 5 1 \
   --sa_to_ffn to_sa \
   --use_varif VFL \
   --use_enc_varif VFL \
   --o2o_matchtype hard \
   --positive_fraction 1.0 \
   --with_aux_class_embed \
   --cls_loss_coef 1 \
   --o2m_cls_loss_coef 0.5 \
   --enc_cls_loss_coef 1 \
   --enc_bbox_loss_coef 5 \
   --enc_giou_loss_coef 2 \
   --topk_eval 300 \
   > $EXP_DIR/train.log