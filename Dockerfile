FROM registry.dp.tech/dptech/deepmd-kit:3.1.1

ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Shanghai
# 确保 PYTHONPATH 包含当前工作目录
ENV PYTHONPATH=/mcp_server/AI4S-agent-tools:/mcp_server/comp-dart-gitlab:$PYTHONPATH

# 确保用户安装的 bin 在系统路径最前面
ENV PATH=/root/.local/bin:$PATH

# UV 配置
ENV UV_PYTHON_INSTALL_MIRROR=https://ghfast.top/github.com/indygreg/python-build-standalone/releases/download
RUN curl -LsSf https://gitee.com/wangnov/uv-custom/releases/download/latest/uv-installer-custom.sh | sh

# 创建目录
RUN mkdir -p /mcp_server/comp-dart-gitlab /root/.dpdispatcher/dp_cloud_server

# 复制文件
COPY . /mcp_server/comp-dart-gitlab
WORKDIR /mcp_server/comp-dart-gitlab

# -----------------------------------------------------------------------------
# 1. 核心构建环境准备
# -----------------------------------------------------------------------------
# 升级 pip 并安装构建后端所需的工具
# scikit-build, cmake, ninja 是 DeepMD-kit 编译的硬依赖
RUN pip install --upgrade pip setuptools wheel scikit-build cmake ninja packaging distro

# -----------------------------------------------------------------------------
# 2. 核心科学计算库 (顺序敏感)
# -----------------------------------------------------------------------------
# 必须先锁定 NumPy 版本，防止 TF/Torch 自动拉取不兼容的新版
RUN pip install numpy==1.26.4

# 安装深度学习框架 (按你要求使用 TF 2.20.0)
RUN pip install tensorflow==2.20.0
RUN pip install torch==2.7.0 torchvision torchaudio

# -----------------------------------------------------------------------------
# 3. 安装 DeepMD-kit (修复 BackendUnavailable 问题)
# -----------------------------------------------------------------------------
# 切换到临时目录进行源码编译，解决 pip 远程安装时的路径解析错误
WORKDIR /tmp/deepmd_build

# 克隆指定 commit
RUN git clone https://github.com/iProzd/deepmd-kit.git . \
    && git checkout 4cc677d6adf4fa1fd6202fbc6008bbd6bd0fe21f

# 调试：验证构建脚本是否可被 Python 识别 (这一步能确保环境正常)
RUN python -c "import sys; sys.path.append('.'); import backend.dp_backend; print('Check: Backend imported successfully')"

# 执行安装
# -v: 显示编译详情
# --no-build-isolation: 使用已安装的 TF 和 scikit-build
# --no-deps: 避免 pip 再次检查依赖导致版本冲突 (我们已经手动管理了核心依赖)
RUN pip install -v --no-build-isolation --no-deps .

# -----------------------------------------------------------------------------
# 4. 后续依赖与清理
# -----------------------------------------------------------------------------
# 回到工作目录
WORKDIR /mcp_server/comp-dart-gitlab
RUN rm -rf /tmp/deepmd_build

# 安装应用层依赖
RUN pip install pymatgen ase pyyaml jsonpickle>=4.1.1 bohr-agent-sdk>=0.1.101
RUN pip install "dpdispatcher @ git+https://github.com/zjgemi/dpdispatcher.git@sandbox"
RUN pip install "bohrium-sdk @ git+https://github.com/zjgemi/bohrium-openapi-python-sdk.git@sandbox-env"

# -----------------------------------------------------------------------------
# 5. 最终验证
# -----------------------------------------------------------------------------
RUN python -c "import deepmd; print(f'DeepMD installed: {deepmd.__file__}')"
RUN python -c "import comp_dart; print('Successfully imported comp_dart')"
RUN python -c "import torch; print('Successfully imported torch')"