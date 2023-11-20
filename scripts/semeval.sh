CUDA_VISIBLE_DEVICES=3 python main.py --max_epochs=12  --num_workers=8 \
    --model_name_or_path PLMs/roberta-large \
    --accumulate_grad_batches 1 \
    --batch_size 16 \
    --data_dir dataset/semeval \
    --check_val_every_n_epoch 1 \
    --data_class WIKI80 \
    --max_seq_length 256 \
    --model_class RobertaForPrompt \
    --t_lambda 0.001 \
    --wandb \
    --litmodel_class BertLitModel \
    --task_name wiki80 \
    --lr 3e-5 \
    --prompt_model PGN_Prompt

# 90.2
'''
加入了concpet graph的效果
wandb: Run summary:
wandb:        Eval/best_f1 0.90004
wandb:             Eval/f1 0.90004
wandb:           Eval/loss 0.66283
wandb:             Test/f1 0.90318
wandb:       Train/ke_loss 0.01704
wandb:          Train/loss 0.01704
wandb:               epoch 9
wandb: trainer/global_step 4070
'''