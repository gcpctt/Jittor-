# 环境安装
根据environment.yaml安装环境

# 数据集下载
将Jittor官网下载的数据放到data/data_A或B中，根据使用的是A榜或者B榜数据

# Train
运行train.sh脚本，需要给出5个参数，第一个是TRAIN_BATCH_SIZE，第二个是MAX_TRAIN_STEPS，第三个是RANK(文本指导分数)，第四个是GPU_VISIBLE，第五个是METHOD
生成的训练权重保存在"checkpoint/style_METHOD_TRAIN_BATCH_SIZE_MAX_TRAIN_STEPS_RANK"中。

# Test
运行test.sh脚本。其中的output参数是图片输出目录，style_path参数是权重加载路径。
