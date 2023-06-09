# 蛋白质超家族分类预测

这个项目方便参赛者测试docker镜像是否正确提交，以及提供测试代码的参考，可以根据自己的需求对测试代码进行修改。

## 文件说明
```
/code
    CustomizeDataset.ipynb, 使用torchdrug库构建自己的数据集，并基于GearNet的demo
    test.py, 将前面的notebook改成py文件
    test_pretrain.py, 在前一个基础上加上pretrain
    testTorchDrug.ipynb, torchdrug的示例代码
    testTorchProt.ipynb，torchprotein的示例代码
 
 env.yaml, 导出的conda环境配置
           如果要导入环境配置可以使用命令：conda env create -f env.yaml
 
```

**数据集**
改用不同的数据集应该修改`/code/test.py`中的`ScopeSuperFamilyClassify`类\
属性：
    `source_file`
    ``\
以及初始化类时传入的`path`
```
pdb_40数据集
path = '/home/tangwuguo/datasets/scope40'
pdb_dir = '/home/tangwuguo/datasets/scope40/dbstyle_all-40-2.08'
source_file = '/home/tangwuguo/datasets/scope40/pdb_scope_db40.csv'

从pdb_40中采的200小数据集
path = '/home/tangwuguo/datasets/scope40_s'
pdb_dir = '/home/tangwuguo/datasets/scope40_s/ents'
source_file = '/home/tangwuguo/datasets/scope40_s/scop40_s.csv'

```

## 如何使用

### 1. 安装依赖

```bash
bash downloads.sh
```

### 2. 拉取镜像并运行

```bash
export user=<your username>
export passwd=<your password>
export image=<your image url>
bash run.sh $user $passwd $image
```

### 3. 测试

```bash
python test.py datas/in.lst datas/out.lst
```

## 参考文件

- docker提交相关

    `docs/CBC Challenge 2023阿里云版参赛手册V2.1.1-byC.pdf`

- 比赛题目

    `docs/蛋白质超家族分类预测.pdf`
