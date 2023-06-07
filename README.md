# 蛋白质超家族分类预测

这个项目方便参赛者测试docker镜像是否正确提交，以及提供测试代码的参考，可以根据自己的需求对测试代码进行修改。

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