# 使用官方镜像，内含 deepmd-kit, tensorflow, cuda 环境
FROM registry.dp.tech/dptech/deepmd-kit:3.1.0-cuda12.1

ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Shanghai
# 确保 import 能找到你的代码
ENV PYTHONPATH=/mcp_server/comp-dart-gitlab:$PYTHONPATH
ENV PATH=/root/.local/bin:$PATH

WORKDIR /mcp_server/comp-dart-gitlab
COPY . .

# 1. 安装 UV
ENV UV_PYTHON_INSTALL_MIRROR=https://ghfast.top/github.com/indygreg/python-build-standalone/releases/download
RUN curl -LsSf https://gitee.com/wangnov/uv-custom/releases/download/latest/uv-installer-custom.sh | sh

# 2. 使用 uv 安装依赖 (比 pip 快且稳)
# 注意：使用 --system 安装到系统 Python，不创建 venv
# 注意：排除 deepmd-kit，防止覆盖镜像自带版本
RUN uv pip install --system --no-deps .

# 3. 补充安装 pyproject.toml 里写了但在 Base Image 里可能缺少的包
# 或者直接让 uv 读取 pyproject.toml 安装除 deepmd 以外的包
# 这里手动列出关键包以确保安全
RUN uv pip install --system \
    pymatgen ase pyyaml "numpy==1.26.4" "jsonpickle>=4.1.1" "bohr-agent-sdk>=0.1.101" \
    "dpdispatcher @ git+https://github.com/zjgemi/dpdispatcher.git@sandbox" \
    "bohrium-sdk @ git+https://github.com/zjgemi/bohrium-openapi-python-sdk.git@sandbox-env"

# 4. 验证
RUN python -c "import deepmd; print(f'DeepMD version: {deepmd.__version__}')"
RUN python -c "import comp_dart; print('Successfully imported comp_dart')"