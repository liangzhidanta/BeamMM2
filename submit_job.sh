#! /bin/bash
### 表示这是一个bash脚本

#SBATCH --job-name=BEAMMM2_TEST
### 设置该作业的作业名

#SBATCH -N 1

#SBATCH -n 8

#SBATCH --gres=gpu:1

#SBATCH --time=24:00:00
### 作业最大的运行时间，超过时间后作业资源会被SLURM回收

#SBATCH --comment llm_6g
### 指定从哪个项目扣费。如果没有这条参数，则从个人账户扣费

#SBATCH --partition=g078t2

source ~/.bashrc
### 初始化环境变量

source activate bind
cd /groups/g900403/home/share/wzj/wyh/BeamMM2
sh train_test.sh
### 程序的执行命令