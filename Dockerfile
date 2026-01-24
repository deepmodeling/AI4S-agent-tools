FROM registry.dp.tech/dptech/deepmd-kit:3.1.0-cuda12.1

ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Shanghai
# [关键] 将当前目录加入 PYTHONPATH，这样就可以直接 import comp_dart
ENV PYTHONPATH=/mcp_server/comp-dart-gitlab:$PYTHONPATH
ENV PATH=/root/.local/bin:$PATH

WORKDIR /mcp_server/comp-dart-gitlab
COPY . .

# 1. 安装 UV (工具链)
ENV UV_PYTHON_INSTALL_MIRROR=https://ghfast.top/github.com/indygreg/python-build-standalone/releases/download
RUN curl -LsSf https://gitee.com/wangnov/uv-custom/releases/download/latest/uv-installer-custom.sh | sh

# 2. 手动安装应用层依赖 (Dependencies)
# 既然不读 pyproject.toml，我们就在这里显式列出依赖
# 注意：排除 deepmd-kit, tensorflow, torch (Base Image 已自带)
RUN uv pip install --system --no-deps \
    tqdm "requests>=2.32.3" "flask>=3.1.1" \
    "scipy>=1.12.0" "ase>=3.22.1" "seekpath>=2.0.1" \
    "numpy==1.26.4" "dpdata==0.2.25" \
    phonopy pymatgen spglib matplotlib typing-extensions pyyaml \
    "bohr-agent-sdk>=0.1.101" "jsonpickle>=4.1.1"

# 3. 安装 Git 依赖 (应用层)
RUN uv pip install --system --no-deps \
    "dpdispatcher @ git+https://github.com/zjgemi/dpdispatcher.git@sandbox" \
    "bohrium-sdk @ git+https://github.com/zjgemi/bohrium-openapi-python-sdk.git@sandbox-env"

# 4. 验证
# 只要 import 成功，说明 PYTHONPATH 设置正确，完全不需要 pip install .
RUN python -c "import comp_dart; print('Successfully imported comp_dart')"
RUN python -c "import deepmd; print(f'DeepMD version: {deepmd.__version__}')"