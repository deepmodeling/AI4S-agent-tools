FROM registry.dp.tech/dptech/deepmd-kit:3.1.1

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
# 1. 首先安装构建工具 (这是 --no-build-isolation 必须的前置条件)
# setuptools, scikit-build, cmake, ninja 是编译 DeepMD-kit C++ 代码所必需的
RUN pip install --upgrade pip setuptools wheel scikit-build cmake ninja

# 2. 提前锁定 NumPy 版本（防止 TF 安装新版后被降级导致损坏）
# 注意：请确认 TF 2.20 是否兼容 NumPy 1.26。如果不兼容，这里会报错或导致 TF 不可用。
RUN pip install numpy==1.26.4

# 3. 安装 TensorFlow 和 PyTorch
# 建议：如果编译 DeepMD 报错，请尝试降低 TF 版本（如 2.15 或 2.12），除非你确定该分支适配了 TF 2.20
RUN pip install tensorflow==2.20.0
RUN pip install torch==2.7.0 torchvision torchaudio

# 4. 安装 DeepMD-kit
# 此时环境中有 TF 和 scikit-build，--no-build-isolation 才能成功
RUN pip install --no-build-isolation git+https://github.com/iProzd/deepmd-kit.git@4cc677d6adf4fa1fd6202fbc6008bbd6bd0fe21f

# 5. 安装其余依赖
# 注意：bohr-agent-sdk 可能会再次检查 numpy 版本，确保兼容
RUN pip install pymatgen ase pyyaml jsonpickle>=4.1.1 bohr-agent-sdk>=0.1.101
RUN pip install "dpdispatcher @ git+https://github.com/zjgemi/dpdispatcher.git@sandbox"
RUN pip install "bohrium-sdk @ git+https://github.com/zjgemi/bohrium-openapi-python-sdk.git@sandbox-env"
# RUN pip install . --no-deps
# RUN uv clean

RUN python -c "import comp_dart; print('Successfully imported comp_dart')"
RUN python -c "import torch; print('Successfully imported torch')"
# RUN uv run python -c "import comp_dart; print('Successfully imported comp_dart')"
# RUN uv run python -c "import torch; print('Successfully imported torch')"