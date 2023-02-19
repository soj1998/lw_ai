# 2023年1月9日 创建项目
+ 电脑建立文件夹，github创建项目
+ conda创建虚拟环境，指定python编译器  
conda create -n lw_ai python=3.8
+ 电脑文件夹下运行git命令，github有提示  
如git add、commit、remote git和push等
+ pycharm不显示环境，可以修改  
设置setting-Tools-Terminal-Shell Path,  
改为C:\Windows\system32\cmd.exe
+ pip与pip3的区别  
如果系统中只安装了Python2，那么就只能使用pip。  
如果系统中只安装了Python3，那么既可以使用pip也可以使用pip3，  
二者是等价的。  
如果系统中同时安装了Python2和Python3，  
则pip默认给Python2用，pip3指定给Python3用。
+ 导入numpy  
在pycharm终端切换到lw_ai，再运行 
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy
+ 运行numpy语句
+ 安装scikit-learn
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scikit-learn

# 2023年1月11日 对数几率回归
安装matplotlib 
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple matplotlib

# 2023年2月10日 tushare
pip install tushare -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2023年2月10日 akshare 李沐相关
https://www.akshare.xyz/ 
https://github.com/akfamily/akshare
pip install akshare --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple
d2l pytorch
pip install d2l -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple