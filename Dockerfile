FROM registry.dp.tech/dptech/deepmd-kit:3.1.1

ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Shanghai
# 确保 PYTHONPATH 包含当前目录，避免 import 路径问题
ENV PYTHONPATH=/mcp_server/AI4S-agent-tools:/mcp_server/comp-dart-gitlab:$PYTHONPATH

# 关键：确保用户安装目录在 PATH 最前面，且 pip 能找到
ENV PATH=/root/.local/bin:$PATH

ENV UV_PYTHON_INSTALL_MIRROR=https://ghfast.top/github.com/indygreg/python-build-standalone/releases/download
RUN curl -LsSf https://gitee.com/wangnov/uv-custom/releases/download/latest/uv-installer-custom.sh | sh

RUN mkdir -p /mcp_server/comp-dart-gitlab /root/.dpdispatcher/dp_cloud_server

COPY . /mcp_server/comp-dart-gitlab
WORKDIR /mcp_server/comp-dart-gitlab

# 1. 安装构建工具和基础依赖
# 使用 --upgrade 确保 pip 本身是最新的，能更好处理 wheel
RUN pip install --upgrade pip setuptools wheel scikit-build cmake ninja

# 2. 安装数值计算核心库
# 锁定 Numpy 防止 ABI 不兼容
RUN pip install numpy==1.26.4

# 3. 安装深度学习框架
# 警告：如果 TF 2.20.0 是内部版本，请确保它与 Base Image 的 CUDA (通常是 11.8 或 12.x) 兼容
# 如果这是官方源，请降级到 tensorflow==2.16.1 (DeepMD 3.x 常用) 或 2.15.0
RUN pip install tensorflow==2.20.0
RUN pip install torch==2.7.0 torchvision torchaudio

# 4. [关键步骤] 调试检查：在编译 DeepMD 之前验证环境
# 这行命令会告诉你：到底是谁导致了 BackendUnavailable
RUN python -c "import sys; print('Python:', sys.executable); import skbuild; print('Skbuild:', skbuild.__file__); import tensorflow as tf; print('TF Version:', tf.__version__); print('TF Path:', tf.__file__)"

# 5. 安装 DeepMD-kit
# 加上 -v (verbose) 可以看到详细的构建日志，而不是只报一个 BackendUnavailable
RUN pip install -v --no-build-isolation git+https://github.com/iProzd/deepmd-kit.git@4cc677d6adf4fa1fd6202fbc6008bbd6bd0fe21f

# 6. 安装后续依赖
RUN pip install pymatgen ase pyyaml jsonpickle>=4.1.1 bohr-agent-sdk>=0.1.101
RUN pip install "dpdispatcher @ git+https://github.com/zjgemi/dpdispatcher.git@sandbox"
RUN pip install "bohrium-sdk @ git+https://github.com/zjgemi/bohrium-openapi-python-sdk.git@sandbox-env"

# 验证最终安装
RUN python -c "import deepmd; print('DeepMD installed:', deepmd.__file__)"