FROM registry.dp.tech/dptech/deepmd-kit:3.1.0-cuda12.1

ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Shanghai
# 纯 Python 项目直接加路径，无需打包
ENV PYTHONPATH=/mcp_server/comp-dart-gitlab:$PYTHONPATH
ENV PATH=/root/.local/bin:$PATH

# 配置 UV
ENV UV_PYTHON_INSTALL_MIRROR=https://ghfast.top/github.com/indygreg/python-build-standalone/releases/download
RUN curl -LsSf https://gitee.com/wangnov/uv-custom/releases/download/latest/uv-installer-custom.sh | sh

RUN mkdir -p /mcp_server/comp-dart-gitlab /root/.dpdispatcher/dp_cloud_server
COPY . /mcp_server/comp-dart-gitlab
WORKDIR /mcp_server/comp-dart-gitlab

# =============================================================================
# Step 1: 准备构建环境
# =============================================================================
RUN pip uninstall -y deepmd-kit || true

# 安装构建依赖
RUN uv pip install --system --upgrade \
    pip setuptools wheel \
    cmake ninja packaging distro pathspec pyproject_metadata \
    "scikit-build-core>=0.5.0" \
    "setuptools_scm>=8.0" \
    hatch-fancy-pypi-readme

# =============================================================================
# Step 2: 编译 DeepMD-kit (关键修正版)
# =============================================================================
WORKDIR /tmp/deepmd_build

# 1. 克隆分支
RUN git clone -b D0708_dpa3_default_fparam https://github.com/iProzd/deepmd-kit.git .

# 2. [关键] 清洗编译器环境 (Clean Compiler Environment)
# Base Image 设置了 CXXFLAGS/LDFLAGS 指向内部的 compiler_compat，导致链接系统 GLIBC 时崩溃。
# 我们必须清空这些变量，并强制指定使用系统的 GCC/G++。
ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++
ENV CFLAGS=""
ENV CXXFLAGS=""
ENV LDFLAGS=""
# 有些 Conda 镜像会设置 LD_LIBRARY_PATH 干扰连接，建议重置或审慎处理，这里先不清空以免影响 CUDA

# 3. 编译安装
# 这里的 --no-build-isolation 和 --no-deps 依然必不可少
ENV DP_ENABLE_TENSORFLOW=1
RUN uv pip install --system -v --no-build-isolation --no-deps .

# =============================================================================
# Step 3: 安装应用依赖 (修复缺少 orjson 等子依赖的问题)
# =============================================================================
WORKDIR /mcp_server/comp-dart-gitlab
RUN rm -rf /tmp/deepmd_build

# [关键修改]
# 1. 去掉 --no-deps: 让 uv 自动安装 pymatgen 依赖的 orjson, monty, pandas 等
# 2. 显式锁定 numpy==1.26.4: 确保 uv 解析依赖时，不会为了迎合其他包而升级 numpy
# 3. 显式锁定 Flask, Requests 等版本
RUN uv pip install --system \
    "numpy==1.26.4" \
    "tqdm" \
    "requests>=2.32.3" \
    "flask>=3.1.1" \
    "scipy>=1.12.0" \
    "ase>=3.22.1" \
    "seekpath>=2.0.1" \
    "dpdata==0.2.25" \
    "phonopy" \
    "pymatgen" \
    "spglib" \
    "matplotlib" \
    "typing-extensions" \
    "pyyaml" \
    "bohr-agent-sdk>=0.1.101" \
    "jsonpickle>=4.1.1"

# Git 依赖通常包含子依赖，建议也去掉 --no-deps (除非你非常确定它们不缺包)
RUN uv pip install --system \
    "dpdispatcher @ git+https://github.com/zjgemi/dpdispatcher.git@sandbox" \
    "bohrium-sdk @ git+https://github.com/zjgemi/bohrium-openapi-python-sdk.git@sandbox-env"

# =============================================================================
# Step 4: 验证
# =============================================================================
# 验证 pymatgen 及其依赖 orjson 是否正常
RUN python -c "import pymatgen.core; print('Pymatgen imported successfully')"
RUN python -c "import deepmd; print(f'DeepMD version: {deepmd.__version__}')"
RUN python -c "import comp_dart; print('Successfully imported comp_dart')"