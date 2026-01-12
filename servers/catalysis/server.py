import sys
import os
from pathlib import Path
from typing import Dict

# Add parent directory to import server_utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from server_utils import mcp_server, setup_server

import numpy as np
from dflow import Workflow
from dp.agent.server import CalculationMCPServer
import argparse
from adsec.core.workflow import ADSECFlow
from adsec.utils.utils import config_workflow

def parse_args():
    """Parse command line arguments for MCP server."""
    parser = argparse.ArgumentParser(description="DPA Calculator MCP Server")
    parser.add_argument('--port', type=int, default=50001, help='Server port (default: 50001)')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    try:
        args = parser.parse_args()
    except SystemExit:
        class Args:
            port = 50001
            host = '0.0.0.0'
            log_level = 'INFO'
        args = Args()
    return args

args = parse_args()
mcp = CalculationMCPServer("CatalysisMCP", host=args.host, port=args.port)

def check_dirs_structure(work_dir: Path, required_dirs: list):
    for dir_name in required_dirs:
        dir_path = work_dir / dir_name
        poscar_path = dir_path / "POSCAR"

        assert dir_path.exists(), f"Missing required directory: {dir_path}"
        assert poscar_path.exists(), f"Missing POSCAR file in {dir_path}"


def get_energies(config_file, work_dir):
    # 初始化配置和工作流
    config, executor_run = config_workflow(str(config_file), work_dir)
    adsec_flow = ADSECFlow(work_dir, executor_run)

    # 添加计算步骤并提交工作流
    adsec_flow.wf.add(adsec_flow.structure2run_step(surface_low=False))
    adsec_flow.wf.submit()

    print("Waiting for Workflow Finish~")
    adsec_flow.wf.wait()

    # 验证工作流是否成功完成
    assert adsec_flow.wf.query_status() == "Succeeded", "Workflow Error, Please Check!"
    print("Workflow Succeeded, Now PostProcess")

    # 获取计算结果
    wf = Workflow(id=adsec_flow.wf.id)
    wf_info = wf.query()
    steps = wf_info.get_step(key="adsec-run")

    # 提取能量数据
    try:
        energies = {
            step.outputs.parameters["structure_path"].value: float(step.outputs.parameters["energy"].value)
            for step in steps
        }
    except KeyError as e:
        raise KeyError(f"Missing required energy data in results: {e}")

    return energies


@mcp.tool()
def cal_ads_energy(config_file: Path, work_dir: Path) -> Dict[str, float]:
    """计算并返回吸附体系的能量信息，包括吸附能。

    该函数执行以下操作：
    1. 验证工作目录结构是否符合要求
    2. 初始化工作流配置
    3. 设置并运行吸附能计算流程(ADSECFlow)
    4. 等待计算完成并验证结果
    5. 从计算结果中提取能量数据并计算吸附能

    Args:
        config_file: 工作流配置文件的路径
        work_dir: 工作目录路径，必须包含以下子目录，每个子目录中包含POSCAR文件：
        - 'slab': 基底结构(POSCAR)
        - 'adslab': 吸附后体系结构(POSCAR)
        - 'mol': 分子结构(POSCAR)

    Returns:
        包含各能量结果的字典，键值对为：
        - 'adslab energy (eV)': 吸附体系的能量(eV)
        - 'slab energy (eV)': 基底的能量(eV)
        - 'mol energy (eV)': 分子的能量(eV)
        - 'adsorption energy (eV)': 计算得到的吸附能(eV)

    Raises:
        AssertionError: 如果工作流未能成功完成或目录结构不符合要求
        FileNotFoundError: 如果缺少必要的POSCAR文件
        KeyError: 如果计算结果中缺少必要的能量数据
    """

    check_dirs_structure(work_dir=work_dir, required_dirs=['slab', 'adslab', 'mol'])
    energies = get_energies(config_file=config_file, work_dir=work_dir)
    ads_energy = energies["adslab"] - energies["slab"] - energies["mol"]

    return {
        "adslab energy (eV)": energies["adslab"],
        "slab energy (eV)": energies["slab"],
        "mol energy (eV)": energies["mol"],
        "adsorption energy (eV)": ads_energy
    }


@mcp.tool()
def cal_surface_energy(config_file: Path, work_dir: Path) -> Dict[str, float]:
    """计算并返回晶胞和表面的能量信息，包括表面能。

        该函数执行以下操作：
        1. 验证工作目录结构是否符合要求
        2. 初始化工作流配置
        3. 设置并运行表面能计算流程(ADSECFlow)
        4. 等待计算完成并验证结果
        5. 从计算结果中提取能量数据并计算表面能

        Args:
            config_file: 工作流配置文件的路径
            work_dir: 工作目录路径，必须包含以下子目录，每个子目录中包含POSCAR文件：
            - 'bulk': 晶胞结构(POSCAR)
            - 'slab': 表面结构(POSCAR)

        Returns:
            包含各能量结果的字典，键值对为：
            - 'bulk energy (eV)': 吸附体系的能量(eV)
            - 'slab energy (eV)': 表面的能量(eV)
            - 'surface energy (eV)': 计算得到的表面能(eV)

        Raises:
            AssertionError: 如果工作流未能成功完成或目录结构不符合要求
            FileNotFoundError: 如果缺少必要的POSCAR文件
            KeyError: 如果计算结果中缺少必要的能量数据
        """

    check_dirs_structure(work_dir=work_dir, required_dirs=['bulk', 'slab'])

    with open(Path(f"{work_dir}/bulk/POSCAR")) as bulk_POSCAR:
        lines = bulk_POSCAR.readlines()
        atom_num_bulk = sum(map(int, lines[6].split()))

    with open(Path(f"{work_dir}/slab/POSCAR")) as slab_POSCAR:
        lines = slab_POSCAR.readlines()
        atom_num_slab = sum(map(int, lines[6].split()))
        print(atom_num_slab)
        a = np.array(list(map(float, lines[2].split())))
        b = np.array(list(map(float, lines[3].split())))

    n = atom_num_slab / atom_num_bulk
    cross_product = np.cross(a, b)
    area = np.linalg.norm(cross_product)

    energies = get_energies(config_file=config_file, work_dir=work_dir)
    surface_energy = (energies["slab"] - n * energies["bulk"]) / (2 * area)
    print(f"表面能为：{surface_energy}")

    return {
        "bulk energy (eV)": energies["bulk"],
        "slab energy (eV)": energies["slab"],
        "surface energy (eV/Å²)": surface_energy
    }


@mcp.tool()
def cal_vacancy_formation_energy(config_file: Path, work_dir: Path) -> Dict[str, float]:
    """计算并返回氧空位形成能。

    Args:
        config_file: 工作流配置文件的路径
        work_dir: 工作目录路径，必须包含以下子目录，每个子目录中包含POSCAR文件：
        - 'perfect': 完美结构
        - 'defect': 缺陷结构
        - 'O2': 氧气分子结构

    Returns:
        包含能量结果和空位形成能的字典：
        - 'perfect energy (eV)': 完美结构的能量
        - 'defect energy (eV)': 缺陷结构的能量
        - 'O2 energy (eV)': 氧气分子的能量
        - 'vacancy formation energy (eV)': 氧空位形成能

    Raises:
        AssertionError: 如果目录结构或工作流状态不正确
        FileNotFoundError: 如果POSCAR文件缺失
        KeyError: 如果缺少能量数据
    """
    check_dirs_structure(work_dir=work_dir, required_dirs=['perfect', 'defect', 'O2'])
    energies = get_energies(config_file=config_file, work_dir=work_dir)
    evf = energies["defect"] + 0.5 * energies["O2"] - energies["perfect"]
    print(f"vacancy formation energy = {evf} eV")

    return {
        "perfect energy (eV)": energies["perfect"],
        "defect energy (eV)": energies["defect"],
        "O2 energy (eV)": energies["O2"],
        "vacancy formation energy (eV)": evf
    }

if __name__ == '__main__':
    # Get transport type from environment variable, default to SSE
    transport_type = os.getenv('MCP_TRANSPORT', 'sse')
    mcp.run(transport=transport_type)