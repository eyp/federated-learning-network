FROM python:3.8-slim

COPY . /federated-learning/

WORKDIR /federated-learning/

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update --fix-missing && \
apt-get install -y build-essential && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*

RUN pip install fastai==2.0.16 torch==1.6 torchvision==0.7 \
&& pip install python-dotenv \
&& pip install aiohttp[speedups] \
&& pip install flask

RUN apt-get purge -y --auto-remove build-essential

EXPOSE 5000
CMD ["/usr/local/bin/flask", "run"]



