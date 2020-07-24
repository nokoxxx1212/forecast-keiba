# forecast-keiba
競馬予想のプロジェクトです。
Python, Docker, Kedro, MLflowを使っています。

## Getting Started for Developer
```
# set google-chrome-stable_current_amd64.deb
https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
->
docker/pytorch_1_4/google-chrome-stable_current_amd64.deb

# build container
$ docker build ./ -t forecast-keiba

# run container
$ sh run.sh docker

# [optional]run container which use gpu(nvidia-docker)
$ sh run.sh nvidia-docker

# [optional]install jupyter extensions
$ pip3 install jupyter-tabnine
$ jupyter nbextension install --py jupyter_tabnine
$ jupyter nbextension enable --py jupyter_tabnine
$ jupyter serverextension enable --py jupyter_tabnine

# start jupyter notebook
$ jupyter notebook --allow-root --port=8888 --ip=0.0.0.0 &

# exec pipeline
$ kedro run

# setup mlflow
$ mlflow ui --host 0.0.0.0 --backend-store-uri logs/mlruns
```

## Description
* OS version
```
$ cat /etc/issue
Ubuntu 18.04.3
```

* Python version
```
$ python -V
Python 3.7.4
```

* PyTorch version
```
>>> torch.__version__
'1.4.0'
```

## Reference
* [競馬予想で始めるデータ分析・機械学習](https://www.youtube.com/channel/UCDzwXAWu1zIfJuPTTZyWthw)