FROM tensorflow/tensorflow:2.8.3-gpu

# Needed for string substitution
SHELL ["/bin/bash", "-c"]
# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl 

ARG PYTHON=python3
ARG PIP=pip3

ENV LANG C.UTF-8

RUN ${PIP} install matplotlib \
  pandas \
  scikit-learn \
  psutil


RUN git clone https://github.com/bzhanglab/AutoRT \
  && mv AutoRT /opt/

COPY bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

WORKDIR /opt/AutoRT/

#specify the command executed when the container is started
CMD ["/bin/bash"]


