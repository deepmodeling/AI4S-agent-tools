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

# RUN uv sync
# Install the package in development mode to make it importable
RUN pip install git+https://github.com/iProzd/deepmd-kit.git@4cc677d6adf4fa1fd6202fbc6008bbd6bd0fe21f
RUN pip install pymatgen ase pyyaml numpy==1.26.4  jsonpickle>=4.1.1 bohr-agent-sdk>=0.1.101
RUN pip install "dpdispatcher @ git+https://github.com/zjgemi/dpdispatcher.git@sandbox"
RUN pip install "bohrium-sdk @ git+https://github.com/zjgemi/bohrium-openapi-python-sdk.git@sandbox-env"
# RUN pip install . --no-deps
# RUN uv clean

RUN python -c "import comp_dart; print('Successfully imported comp_dart')"
RUN python -c "import torch; print('Successfully imported torch')"
# RUN uv run python -c "import comp_dart; print('Successfully imported comp_dart')"
# RUN uv run python -c "import torch; print('Successfully imported torch')"