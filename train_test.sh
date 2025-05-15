#!/bin/bash
# run_models.sh
# 本脚本分别训练两个模型（均使用 MSELoss 作为损失函数），
# 模型之间的区别在于加载不同的预训练编码器权重。
# 训练结束后，分别调用 evaluate_MultiModalEncoderDecoderModel.py 对两个模型进行评估。
#
# 使用方法：
#   chmod +x run_models.sh
#   ./run_models.sh

# 使用CUDA:1
export CUDA_VISIBLE_DEVICES=1

###############################
# 公共参数设置
###############################
LOSS="MSE"
EPOCHS=50
BATCH_SIZE=64
LEARNING_RATE=1e-4
PATIENCE=5

###############################
# 模型1相关参数设置（加载预训练编码器1）
###############################
CHECKPOINT_DIR1="./fig1/imagebind_s6"
PRETRAINED_ENCODER_PATH1="/home/wzj/BeamMM/checkpoints/vision_s6.pth"  # MMWAVE ANCHOR

###############################
# 模型2相关参数设置（加载预训练编码器2）
###############################
CHECKPOINT_DIR2="./checkpoints_model2"
PRETRAINED_ENCODER_PATH2="/groups/g900403/home/share/wzj/BeamMM/checkpoints/mmwave_gps_joint-01-15-152626.pth"   # VISION ANCHOR

###############################
# 确保保存检查点的目录存在
###############################
mkdir -p "$CHECKPOINT_DIR1"
mkdir -p "$CHECKPOINT_DIR2"

# 生成日志文件时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

###############################
# 开始训练模型1
###############################
echo "============================="
echo "开始训练模型1（使用预训练编码器：$PRETRAINED_ENCODER_PATH1）..."
LOG_FILE1="${CHECKPOINT_DIR1}/training_log_${TIMESTAMP}.txt"

python -u train_MultiModalEncoderDecoderModel.py \
    --loss "$LOSS" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --patience "$PATIENCE" \
    --checkpoint_dir "$CHECKPOINT_DIR1" \
    --dataset_start_idx 1 \
    --dataset_end_idx 7 \
    --pretrained_encoder_path "$PRETRAINED_ENCODER_PATH1" | tee "$LOG_FILE1"

echo "模型1训练完成，日志保存在 ${LOG_FILE1}"

# 模型1训练结束后，评估最佳模型（脚本中默认保存最佳模型为 multimodal_encoder_decoder_best.pth）
BEST_MODEL_PATH1="${CHECKPOINT_DIR1}/multimodal_encoder_decoder_best.pth"
if [ -f "$BEST_MODEL_PATH1" ]; then
    echo "开始评估模型1：$BEST_MODEL_PATH1"
    python -u evaluate_MultiModalEncoderDecoderModel.py --model_path "$BEST_MODEL_PATH1"  --dataset_start_idx 1 --dataset_end_idx 7
else
    echo "未找到模型1的最佳权重文件：$BEST_MODEL_PATH1"
fi

###############################
# 开始训练模型2
###############################
echo "============================="
# echo "开始训练模型2（使用预训练编码器：$PRETRAINED_ENCODER_PATH2）..."
# LOG_FILE2="${CHECKPOINT_DIR2}/training_log_${TIMESTAMP}.txt"

# python train_MultiModalEncoderDecoderModel.py \
#     --loss "$LOSS" \
#     --epochs "$EPOCHS" \
#     --batch_size "$BATCH_SIZE" \
#     --learning_rate "$LEARNING_RATE" \
#     --patience "$PATIENCE" \
#     --checkpoint_dir "$CHECKPOINT_DIR2" \
#     --pretrained_encoder_path "$PRETRAINED_ENCODER_PATH2" | tee "$LOG_FILE2"

# echo "模型2训练完成，日志保存在 ${LOG_FILE2}"

# # 模型2训练结束后，评估最佳模型
# BEST_MODEL_PATH2="${CHECKPOINT_DIR2}/multimodal_encoder_decoder_best.pth"
# if [ -f "$BEST_MODEL_PATH2" ]; then
#     echo "开始评估模型2：$BEST_MODEL_PATH2"
#     python evaluate_MultiModalEncoderDecoderModel.py --model_path "$BEST_MODEL_PATH2"
# else
#     echo "未找到模型2的最佳权重文件：$BEST_MODEL_PATH2"
# fi

# echo "============================="
# echo "所有模型训练和评估任务已完成！"
