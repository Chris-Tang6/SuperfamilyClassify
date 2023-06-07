set -e
sudo docker login --username $1 --password $2 $3
sudo docker pull $3
sudo docker run --ipc=host --gpus all  --rm -v `pwd`/datas:/mounted_path $3 bash run.sh /mounted_path/in.lst > datas/out.lst
