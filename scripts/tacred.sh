export CUDA_VISIBLE_DEVICES=0
python main.py --max_epochs=5  --num_workers=8 \
    --model_name_or_path PLMs/roberta-large \
    --accumulate_grad_batches 4 \
    --batch_size 16 \
    --data_dir dataset/tacred/ \
    --check_val_every_n_epoch 1 \
    --data_class WIKI80 \
    --max_seq_length 256 \
    --model_class RobertaForPrompt \
    --t_lambda 0.001 \
    --wandb \
    --litmodel_class BertLitModel \
    --task_name wiki80 \
    --lr 3e-5 \
    --prompt_model  PGN_Prompt \
    --use_template_words 0