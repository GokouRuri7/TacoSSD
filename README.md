
安装运行代码所需要的包

pip3 install -r requirements.txt



如果需要训练网络，需要打开ssd.py
将model_path改为model_data/ssd_weights.pth
如果需要预测图片
将model_path改为logs文件夹中loss最小的文件，我这里选择的是ep098

line 28
"model_path"        : 'model_data/ssd_weights.pth'
"model_path"        : 'logs/ep098-loss1.788-val_loss4.897.pth'



预测图片

python3 predict.py

预测图片的地址为img/photo512
photo512中有502个图片
在预测中输入img/photo512/00000xxx.jpg
xxx为0-501


训练网络

python3 train.py


图片和logs占用的空间太大了，所这我这里进行了删减
完整版地址：
「TacoSSD」，点击链接保存，或者复制本段内容，打开「阿里云盘」APP ，无需下载，极速在线查看享用。
链接：https://www.aliyundrive.com/s/9T6NwgPRq6N

参考代码：https://gitcode.net/mirrors/bubbliiiing/ssd-pytorch?utm_source=csdn_github_accelerator
# TacoSSD
