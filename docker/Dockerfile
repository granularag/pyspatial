FROM ubuntu:xenial
RUN apt-get update && apt-get install -y software-properties-common python-software-properties python-software-properties
COPY install-base.sh /
RUN /install-base.sh
COPY install-python.sh /
RUN /install-python.sh
RUN mkdir /data
RUN mkdir /test
CMD ["jupyter-notebook","--notebook-dir=/notebooks", "--allow-root", "--config=/etc/jupyter/jupyter_notebook_config.py"]
