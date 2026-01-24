FROM registry.dp.tech/dptech/deepmd-kit:3.1.1

ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Shanghai
ENV PYTHONPATH=/mcp_server/AI4S-agent-tools:/mcp_server/comp-dart-gitlab:$PYTHONPATH
ENV PATH=/root/.local/bin:$PATH

ENV UV_PYTHON_INSTALL_MIRROR=https://ghfast.top/github.com/indygreg/python-build-standalone/releases/download
RUN curl -LsSf https://gitee.com/wangnov/uv-custom/releases/download/latest/uv-installer-custom.sh | sh

RUN mkdir -p /mcp_server/comp-dart-gitlab /root/.dpdispatcher/dp_cloud_server

COPY . /mcp_server/comp-dart-gitlab
WORKDIR /mcp_server/comp-dart-gitlab

# -----------------------------------------------------------------------------
# 1. 核心构建环境准备
# -----------------------------------------------------------------------------
# 依然保留 scikit-build-core，因为 DeepMD 新版确实需要它
RUN pip install --upgrade pip setuptools wheel \
    scikit-build scikit-build-core \
    cmake ninja packaging distro pathspec pyproject_metadata

# -----------------------------------------------------------------------------
# 2. 核心科学计算库 (关键修改)
# -----------------------------------------------------------------------------
RUN pip install numpy==1.26.4

# [修改点]：降级到 2.15.1
# TF 2.15 是 DeepMD 旧版构建脚本能原生识别的最后一个版本。
# 这样可以避免 "Failed to read TF version" 错误，且无需修改源码。
RUN pip install tensorflow==2.15.1

RUN pip install torch==2.7.0 torchvision torchaudio

# -----------------------------------------------------------------------------
# 3. 安装 DeepMD-kit
# -----------------------------------------------------------------------------
WORKDIR /tmp/deepmd_build

RUN git clone https://github.com/iProzd/deepmd-kit.git . \
    && git checkout 4cc677d6adf4fa1fd6202fbc6008bbd6bd0fe21f

# 调试：验证导入。
# 因为版本匹配，这一步现在应该能自然通过，不需要 sed hack。
RUN python -c "import sys; sys.path.append('.'); import backend.dp_backend; print('Check: Backend imported successfully')"

# 安装
RUN pip install -v --no-build-isolation --no-deps .

# -----------------------------------------------------------------------------
# 4. 后续依赖与清理
# -----------------------------------------------------------------------------
WORKDIR /mcp_server/comp-dart-gitlab
RUN rm -rf /tmp/deepmd_build

RUN pip install pymatgen ase pyyaml jsonpickle>=4.1.1 bohr-agent-sdk>=0.1.101
RUN pip install "dpdispatcher @ git+https://github.com/zjgemi/dpdispatcher.git@sandbox"
RUN pip install "bohrium-sdk @ git+https://github.com/zjgemi/bohrium-openapi-python-sdk.git@sandbox-env"

# -----------------------------------------------------------------------------
# 5. 最终验证
# -----------------------------------------------------------------------------
RUN python -c "import deepmd; print(f'DeepMD installed: {deepmd.__file__}')"
RUN python -c "import comp_dart; print('Successfully imported comp_dart')"
RUN python -c "import torch; print('Successfully imported torch')"