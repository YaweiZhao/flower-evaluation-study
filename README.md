# 环境要求：

python >= 3.8

pytorch >= 1.11+cu115



# 环境配置：

pip install flwr

pip install torchvision

pip install opencv-python



# 数据存放：

`./data/`（用于测试的分类图像数据，第一级子目录为医院，即客户端，第二级子目录为分类标签）



# 配置文件：

`./config/DenseNet.json`

`{`
    `"data":"./data/01_a/",`
    `"bench_param":{`
        `"server_address": "localhost:8090",`
        `"device": "cuda:1",`
		`"num_rounds": 3`
    `},`
    `"training_param": {`
        `"epochs": 20,`
        `"batch_size": 32,`

​		`"resize": [32, 32]`,

​        `"learning_rate": 0.001,`
​        `"loss_func": "cross_entropy",`
​        `"optimizer": "sgd",`
​        `"optimizer_param": {`
​            `"momentum": 0.9,`
​			`"dampening": 0,`
​            `"weight_decay": 0,`
​            `"nesterov": false`
​        `}`
​    `}`
`}`

`data`：数据集路径

`bench_param`：联邦学习参数。

​	`server_address`为服务器IP地址；

​	`device`为训练设备，可选`cuda:0`或`cuda:1`，`cuda:…`或`cup`，`num_rounds`为服务器+客户端数量（客户	端数量即`num_rounds` - 1）

`training_param`：训练参数。

​	`epochs`为训练轮次；

​	`batch_size`为一次训练塞入的图片数量，越大越占用显存。

​	`resize`为训练初始分辨率，越大图像信息越丰富，同时越占用显存，训练越慢。

​	`learning_rate`为学习率；

​	`loss_func`为训练损失，当前仅支持交叉熵损失和MSE损失，选择`cross_entropy`为交叉熵损失，其他选项或	不填对应`MSE`损失；

​	`optimizer`为优化器，可选`sgd`和`adam`

​	`optimizer_param`优化器参数；



# 快速训练和测试：

`sh run.sh`

训练完成后模型保存在save_model目录下



# 正常训练和测试：

在服务器上执行`python server.py --config=“./config/DenseNet.json”`

在每台客户端上执行`python client.py --config=“./config/DenseNet.json”`

参数可以在`config`下的`DenseNet.json`中修改，**也可以选择用默认参数直接**：

在服务器上执行`python server.py`

在每台客户端上执行`python client.py`

训练完成后模型保存在save_model目录下# flower-evaluation-study
