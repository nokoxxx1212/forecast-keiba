FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

ENV GIT_SSL_NO_VERIFY=1

# apt install
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
         curl \
         git \
         groff-base \ 
         vim \
         wget

# pip install
COPY requirements.txt /tmp/
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

# cd
WORKDIR /opt

# expose port for jupyter notebook 
EXPOSE 8888

# expose port for kedro viz
EXPOSE 4141
