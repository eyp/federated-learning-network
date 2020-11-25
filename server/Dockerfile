FROM python:3.8-slim

COPY . /federated-learning-network/

WORKDIR /federated-learning-network/

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update --fix-missing && \
apt-get install -y build-essential && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*

RUN pip install torch torchvision \
&& pip install fastai \
&& pip install python-dotenv \
&& pip install aiohttp[speedups] \
&& pip install flask

RUN apt-get purge -y --auto-remove build-essential

CMD ["/usr/local/bin/flask", "run"]



