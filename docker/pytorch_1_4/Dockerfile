FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

ENV GIT_SSL_NO_VERIFY=1

# apt install
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
         curl \
         git \
         gnupg2 \
         groff-base \
         unzip \
         vim \
         wget

# pip install
COPY requirements.txt /tmp/
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

# setup selenium
# RUN wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
COPY google-chrome-stable_current_amd64.deb /workspace/
RUN dpkg -i --force-depends google-chrome-stable_current_amd64.deb || true && \
    apt update && \
    apt -f install -y && \
    dpkg -i --force-depends google-chrome-stable_current_amd64.deb && \
    apt install python3-selenium -y && \
    apt install libgconf2-4 -y

# cd
WORKDIR /opt

# expose port for jupyter notebook 
EXPOSE 8888

# expose port for kedro viz
EXPOSE 4141
