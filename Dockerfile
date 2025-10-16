FROM registry.dp.tech/dptech/deepmd-kit:3.1.0-cuda12.1

ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Shanghai
ENV PYTHONPATH=/mcp_server/AI4S-agent-tools:$PYTHONPATH

ENV PATH=/root/.local/bin:$PATH
ENV UV_PYTHON_INSTALL_MIRROR=https://ghfast.top/github.com/indygreg/python-build-standalone/releases/download
RUN curl -LsSf https://gitee.com/wangnov/uv-custom/releases/download/latest/uv-installer-custom.sh | sh

RUN mkdir -p /mcp_server/comp-dart-gitlab /root/.dpdispatcher/dp_cloud_server

COPY . /mcp_server/comp-dart-gitlab
WORKDIR /mcp_server/comp-dart-gitlab

RUN uv sync
# Install the package in development mode to make it importable
# RUN pip install . 
RUN uv clean

RUN python -c "import comp_dart; print('Successfully imported comp_dart')"
RUN python -c "import torch; print('Successfully imported torch')"
# RUN uv run python -c "import comp_dart; print('Successfully imported comp_dart')"
# RUN uv run python -c "import torch; print('Successfully imported torch')"