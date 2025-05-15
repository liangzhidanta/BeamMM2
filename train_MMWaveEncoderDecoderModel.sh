#!/bin/bash

# train.sh
# Bash脚本，用于训练MMWaveEncoderDecoderModel，支持选择MSE或NMSE作为损失函数。

# 使用说明：
# ./train.sh [--loss LOSS] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--learning_rate LR] [--checkpoint_dir CHECKPOINT_DIR] [--pretrained_encoder_path PRETRAINED_ENCODER_PATH]

# 默认参数设置
LOSS="MSE"
EPOCHS=500
BATCH_SIZE=16
LEARNING_RATE=1e-4
PATIENCE=10
CHECKPOINT_DIR="./general_checkpoints_1_8"
PRETRAINED_ENCODER_PATH="/home/wzj/BeamMM/checkpoints/mmwave_gps_joint-01-15-152626.pth"  # 请替换为实际路径

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --loss) LOSS="$2"; shift ;;
        --epochs) EPOCHS="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --learning_rate) LEARNING_RATE="$2"; shift ;;
        --patience) PATIENCE="$2"; shift ;;
        --checkpoint_dir) CHECKPOINT_DIR="$2"; shift ;;
        --pretrained_encoder_path) PRETRAINED_ENCODER_PATH="$2"; shift ;;
        *) echo "未知参数：$1"; exit 1 ;;
    esac
    shift
done

# 创建检查点保存目录（如果不存在）
mkdir -p "$CHECKPOINT_DIR"

# 生成日志文件名，包含时间戳
LOG_FILE="${CHECKPOINT_DIR}/training_log_$(date +%Y%m%d_%H%M%S).txt"

# 打印训练配置信息
echo "训练配置："
echo "损失函数类型：$LOSS"
echo "训练轮数：$EPOCHS"
echo "批量大小：$BATCH_SIZE"
echo "学习率：$LEARNING_RATE"
echo "早停步数：$PATIENCE"
echo "检查点保存目录：$CHECKPOINT_DIR"
echo "预训练编码器权重路径：$PRETRAINED_ENCODER_PATH"
echo "日志文件：$LOG_FILE"
echo "----------------------------------------"

# 运行训练脚本，并将输出记录到日志文件
python train_MMWaveEncoderDecoderModel.py \
    --loss "$LOSS" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --patience "$PATIENCE" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --pretrained_encoder_path "$PRETRAINED_ENCODER_PATH" \
    | tee "$LOG_FILE"

# 训练完成提示
echo "训练完成！日志记录在 $LOG_FILE"
