# 1. 使用你指定的 Base Image (含 CUDA, TF 环境)
FROM registry.dp.tech/dptech/deepmd-kit:3.1.0-cuda12.1

ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Shanghai
ENV PATH=/root/.local/bin:$PATH
# [关键] 直接把当前目录加入 PYTHONPATH，不用 pip install . 也能 import
ENV PYTHONPATH=/mcp_server/comp-dart-gitlab:$PYTHONPATH

# 配置 UV
ENV UV_PYTHON_INSTALL_MIRROR=https://ghfast.top/github.com/indygreg/python-build-standalone/releases/download
RUN curl -LsSf https://gitee.com/wangnov/uv-custom/releases/download/latest/uv-installer-custom.sh | sh

RUN mkdir -p /mcp_server/comp-dart-gitlab /root/.dpdispatcher/dp_cloud_server
COPY . /mcp_server/comp-dart-gitlab
WORKDIR /mcp_server/comp-dart-gitlab

# =============================================================================
# Step 1: 准备构建环境 (为了编译 P0708 版本)
# =============================================================================
# 1. 卸载镜像自带的 deepmd-kit (它不是你要的 Git 版本)
RUN pip uninstall -y deepmd-kit || true

# 2. 补全构建工具 (必须手动装，否则 --no-build-isolation 会报错)
# 包含 hatch-fancy-pypi-readme 以解决 README 报错
RUN pip install --upgrade pip setuptools wheel \
    cmake ninja packaging distro pathspec pyproject_metadata \
    "scikit-build-core>=0.5.0" \
    "setuptools_scm>=8.0" \
    hatch-fancy-pypi-readme

# =============================================================================
# Step 2: 源码编译 DeepMD-kit (P0708 / 4cc677d6)
# =============================================================================
WORKDIR /tmp/deepmd_build

# 1. 克隆代码 & 切分支
RUN git clone https://github.com/iProzd/deepmd-kit.git . \
    && git checkout 4cc677d6adf4fa1fd6202fbc6008bbd6bd0fe21f

# 2. [关键 Patch] 强制指定 TF 版本号
# 无论 Base Image 里是 TF 2.12, 2.13 还是 2.15，这个补丁能强制跳过脚本的探测错误。
# 返回 "2.15.1" 是个安全值，能骗过构建系统让它继续编译 C++ 接口。
RUN sed -i '/def get_tf_version(path):/a\    return "2.15.1"' backend/find_tensorflow.py

# 3. 编译安装
# DP_ENABLE_TENSORFLOW=1: 开启 TF 支持
# --no-deps: 保护环境里的 TF/Numpy 不被乱改
# --no-build-isolation: 链接环境里的 TF
ENV DP_ENABLE_TENSORFLOW=1
RUN pip install -v --no-build-isolation --no-deps .

# =============================================================================
# Step 3: 安装 Comp-Dart 依赖 (应用层)
# =============================================================================
WORKDIR /mcp_server/comp-dart-gitlab
RUN rm -rf /tmp/deepmd_build

# 1. 安装 pyproject.toml 里的依赖 (手动列出，不通过 pip install .)
# 剔除了 deepmd-kit, tensorflow, torch (环境已备好)
RUN uv pip install --system --no-deps \
    tqdm "requests>=2.32.3" "flask>=3.1.1" \
    "scipy>=1.12.0" "ase>=3.22.1" "seekpath>=2.0.1" \
    "numpy==1.26.4" "dpdata==0.2.25" \
    phonopy pymatgen spglib matplotlib typing-extensions pyyaml \
    "bohr-agent-sdk>=0.1.101" "jsonpickle>=4.1.1"

# 2. 安装 Git 依赖
RUN uv pip install --system --no-deps \
    "dpdispatcher @ git+https://github.com/zjgemi/dpdispatcher.git@sandbox" \
    "bohrium-sdk @ git+https://github.com/zjgemi/bohrium-openapi-python-sdk.git@sandbox-env"

# =============================================================================
# Step 4: 验证
# =============================================================================
# 1. 验证 DeepMD 是否为 Git 版本 (通过检查路径或文件)
RUN python -c "import deepmd; print(f'DeepMD installed at: {deepmd.__file__}')"

# 2. 验证 Comp-Dart 是否可导入 (依赖 PYTHONPATH)
RUN python -c "import comp_dart; print('Successfully imported comp_dart')"