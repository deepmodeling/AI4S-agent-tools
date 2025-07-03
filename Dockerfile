FROM registry.dp.tech/dptech/ubuntu:22.04-py3.10-irkernel-r4.4.1

ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Shanghai
ENV PYTHONPATH=/app:$PYTHONPATH

RUN mkdir -p /app
WORKDIR /app

COPY . /app

RUN pip install . --no-cache-dir
RUN pip install git+https://github.com/dptech-corp/bohr-agent-sdk.git@master
RUN pip install git+https://github.com/dingzhaohan/dpdispatcher.git@master
RUN pip install bohrium-sdk
RUN pip install distro
