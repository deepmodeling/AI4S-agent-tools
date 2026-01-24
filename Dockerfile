# 1. Base Image: 使用官方镜像，保底 CUDA/TF 环境
FROM registry.dp.tech/dptech/deepmd-kit:3.1.0-cuda12.1

ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Shanghai
# [关键] 直接设置 PYTHONPATH，无需对 comp-dart 进行 pip install
ENV PYTHONPATH=/mcp_server/comp-dart-gitlab:$PYTHONPATH
ENV PATH=/root/.local/bin:$PATH

# 配置 UV (加速依赖安装)
ENV UV_PYTHON_INSTALL_MIRROR=https://ghfast.top/github.com/indygreg/python-build-standalone/releases/download
RUN curl -LsSf https://gitee.com/wangnov/uv-custom/releases/download/latest/uv-installer-custom.sh | sh

RUN mkdir -p /mcp_server/comp-dart-gitlab /root/.dpdispatcher/dp_cloud_server
COPY . /mcp_server/comp-dart-gitlab
WORKDIR /mcp_server/comp-dart-gitlab

# =============================================================================
# Step 1: 准备构建环境
# =============================================================================
# 卸载自带 DeepMD
RUN pip uninstall -y deepmd-kit || true

# 安装构建依赖 (必须步骤)
# 即使是 D0708 分支，编译 C++ 扩展依然需要 scikit-build-core 和 cmake
RUN uv pip install --system --upgrade \
    pip setuptools wheel \
    cmake ninja packaging distro pathspec pyproject_metadata \
    "scikit-build-core>=0.5.0" \
    "setuptools_scm>=8.0" \
    hatch-fancy-pypi-readme

# =============================================================================
# Step 2: 安装 DeepMD-kit (Branch: D0708_dpa3_default_fparam)
# =============================================================================
WORKDIR /tmp/deepmd_build

# 1. 克隆指定分支 (无需 sed，相信该分支已适配)
RUN git clone -b D0708_dpa3_default_fparam https://github.com/iProzd/deepmd-kit.git .

# 2. 编译安装
# --no-deps: 防止重装 numpy/tf
# --no-build-isolation: 链接宿主机环境
ENV DP_ENABLE_TENSORFLOW=1
RUN uv pip install --system -v --no-build-isolation --no-deps .

# =============================================================================
# Step 3: 安装应用依赖
# =============================================================================
WORKDIR /mcp_server/comp-dart-gitlab
RUN rm -rf /tmp/deepmd_build

# 安装依赖 (排除 deepmd-kit)
RUN uv pip install --system --no-deps \
    tqdm "requests>=2.32.3" "flask>=3.1.1" \
    "scipy>=1.12.0" "ase>=3.22.1" "seekpath>=2.0.1" \
    "numpy==1.26.4" "dpdata==0.2.25" \
    phonopy pymatgen spglib matplotlib typing-extensions pyyaml \
    "bohr-agent-sdk>=0.1.101" "jsonpickle>=4.1.1"

# Git 依赖
RUN uv pip install --system --no-deps \
    "dpdispatcher @ git+https://github.com/zjgemi/dpdispatcher.git@sandbox" \
    "bohrium-sdk @ git+https://github.com/zjgemi/bohrium-openapi-python-sdk.git@sandbox-env"

# =============================================================================
# Step 4: 验证
# =============================================================================
RUN python -c "import deepmd; print(f'DeepMD installed: {deepmd.__file__}')"
RUN python -c "import comp_dart; print('Successfully imported comp_dart')"