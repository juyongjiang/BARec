# Fine-tuning (Beauty & Phones)
# 100 negative sampling evaluation
# python -u main.py --dataset=Beauty \
#                   --lr=0.001 --maxlen=100 --dropout_rate=0.7 --evalnegsample=100 \
#                   --hidden_units=128 --num_blocks=2 --num_heads=4 \
#                   --reversed_pretrain=1 --aug_traindata=15 --M=18 \
#                   --alpha_coef=1.0 --clip_k=12 \
#                   2>&1 | tee fine_tune_beauty.log

# python -u main.py --dataset=Cell_Phones_and_Accessories \
#                   --lr=0.001 --maxlen=100 --dropout_rate=0.5 --evalnegsample=100 \
#                   --hidden_units=32 --num_blocks=2 --num_heads=2 \
#                   --reversed_pretrain=1 --aug_traindata=17 --M=18 \
#                   --alpha_coef=0.2 --clip_k=12 \
#                   2>&1 | tee fine_tune_phones.log

# Full rank evaluation
CUDA_VISIBLE_DEVICES=0 python -u main.py --dataset=Beauty \
                  --lr=0.001 --maxlen=100 --dropout_rate=0.7 --evalnegsample=-1 \
                  --hidden_units=128 --num_blocks=2 --num_heads=4 \
                  --reversed_pretrain=1 --aug_traindata=15 --M=18 \
                  --alpha_coef=1.0 --clip_k=12 \
                  2>&1 | tee fine_tune_beauty_full_rank_singlecard.log

CUDA_VISIBLE_DEVICES=1 python -u main.py --dataset=Cell_Phones_and_Accessories \
                  --lr=0.001 --maxlen=100 --dropout_rate=0.5 --evalnegsample=-1 \
                  --hidden_units=32 --num_blocks=2 --num_heads=2 \
                  --reversed_pretrain=1 --aug_traindata=17 --M=18 \
                  --alpha_coef=0.2 --clip_k=12 \
                  2>&1 | tee fine_tune_phones_full_rank_singlecard.log