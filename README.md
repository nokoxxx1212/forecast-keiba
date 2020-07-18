# forecast-keiba

## Getting Started for Developer
```
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
