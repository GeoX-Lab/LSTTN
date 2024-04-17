### 如何复现
再main文件中导入相应`/configs`中的CFG
例如：
```
from configs.PEMS04.forecasting import CFG

launch_training(CFG)
```
然后直接运行main文件即可

### easytorch
来自https://github.com/cnstark/easytorch
而不是https://github.com/sraashis/easytorch
安装命令是`pip install easy-torch`

### 路径问题
运行main文件时可能会有路径问题相关的报错，建议直接把相对路径改成绝对路径。
需要改的地方包括：
1. `/configs`每一个文件内的GRAPH_PKL_PATH和CFG.TRAIN.DATA.DIR
2. `/models/lsttn.py`中加载预训练transformer部分，115-122行
3. `/runners/lsttn.py`中加载数据部分，25-31行
