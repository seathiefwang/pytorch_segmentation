# 方案说明（“华为・昇腾杯”AI+遥感影像）

## 运行环境
1. python3
2. pytorch
3. cuda & cudnn
4. opencv-python
4. pillow
5. albumentations
6. tqdm
7. tensorboard

## 方案思路

基于UNet修改的网络结构，使用resnet50作为backbone，loss采用的是cross entropy + dice，梯度优化器使用的是带momentum的SGD，学习率lr设置为0.01，momentum设置为0.9，weight decay设置为1e-4，scheduler是cosinewithrestarts。

## 测试复现
在代码根目录运行 python inferrence_multimodel.py 命令

**参数说明：**

-c 配置文件1

-c2 配置文件2

--models 模型文件列表1

--models2 模型文件列表2

-i 输入图片目录

-o 输出标签目录

**参考命令:**
`python inference_multimodel.py -c=config.json -c2=config.json --models saved/ResUnet/10-15_12-15/checkpoint-epoch80.pth saved/ResUnet/10-15_12-15/checkpoint-epoch70.pth --models2 saved/ResUnet/10-16_12-49/checkpoint-epoch90.pth saved/ResUnet/10-16_18-26/checkpoint-epoch90.pth saved/ResUnet/10-17_12-23/checkpoint-epoch100.pth saved/ResUnet/10-17_12-23/checkpoint-epoch95.pth -i=../image_B -o=../results`

## 模型下载

将模型下载之后解压到代码根目录下的saved目录中

百度网盘下载地址：

链接：https://pan.baidu.com/s/1gDzTMwOojrbdEOOnRpA7-A 
提取码：rtac 