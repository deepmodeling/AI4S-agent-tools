FROM registry.dp.tech/dptech/ubuntu:22.04-py3.10-irkernel-r4.4.1

ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Shanghai
ENV PYTHONPATH=/mcp_server/AI4S-agent-tools:$PYTHONPATH

RUN mkdir -p /mcp_server/AI4S-agent-tools /root/.dpdispatcher/dp_cloud_server
WORKDIR /mcp_server/AI4S-agent-tools

COPY . /mcp_server/AI4S-agent-tools


RUN pip install . --no-cache-dir && \
    pip install git+https://github.com/dptech-corp/bohr-agent-sdk.git@master && \
    pip install git+https://github.com/dingzhaohan/dpdispatcher.git@master && \
    pip install bohrium-sdk && \
    pip install distro
