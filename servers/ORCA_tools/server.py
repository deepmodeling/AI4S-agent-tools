#!/usr/bin/env python3
"""
Example MCP Server using the new simplified pattern.
This demonstrates how to create a new AI4S tool with tools defined at module level.
"""
import os
import logging
import argparse
from pathlib import Path
from typing import Optional, TypedDict, List
import subprocess

from dp.agent.server import CalculationMCPServer

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


ORCA = os.environ.get("ORCA", "orca")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def parse_args():
    """Parse command line arguments for MCP server."""
    parser = argparse.ArgumentParser(description="MCP Server")
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
mcp = CalculationMCPServer("orca_tools", host=args.host, port=args.port)
logging.basicConfig(
    level=args.log_level.upper(),
    format="%(asctime)s-%(levelname)s-%(message)s"
    )


class OrcaResult(TypedDict):
    """Result structure for ORCA calculation"""
    output_file: Path
    gbw_file: Path
    mol_file: Path
    message: str


@mcp.tool()
def run_orca_calculation(
    input_str: str,
    xyz_coordinates: str,
) -> OrcaResult:
    """
    运行 ORCA 计算。

    Args:
        input_str (str): 
            ORCA 输入字符串，包含必要的计算参数和分子结构。

            ```example
            ! BLYP def2-SVP
            * xyz 0 1
            C                  0.00000000    0.00000000   -0.56221066
            H                  0.00000000   -0.92444767   -1.10110537
            H                 -0.00000000    0.92444767   -1.10110537
            O                  0.00000000    0.00000000    0.69618930
            *            
            ```
        xyz_coordinates (str):
            分子结构的 XYZ 坐标字符串，每行格式为 "元素符号 x y z"。

            ```example
            C 0.00000000 0.00000000 -0.56221066
            H 0.00000000 -0.92444767 -1.10110537
            H -0.00000000 0.92444767 -1.10110537
            O 0.00000000 0.00000000 0.69618930
            ```
            
    Returns:
        OrcaResult: 包含以下内容的字典：
            - output_file (Path): ORCA 输出文件的路径。
            - gbw_file (Path): ORCA GBW 文件的路径。
            - mol_file (Path): 分子结构文件的路径。
            - message (str): 成功或错误消息。
    """
    try:
        work_dir = Path(f"orca_calc_{int(os.path.getctime('.'))}")
        if not work_dir.exists():
            work_dir.mkdir(parents=True, exist_ok=True)

        orca_input = work_dir / "calc.inp"
        with open(orca_input, "w") as f:
            f.write(input_str)

        mol_xyz = work_dir / "mol.xyz"
        natoms = len(xyz_coordinates.splitlines())
        with open(mol_xyz, "w") as f:
            f.write(f"{natoms}\n\n")
            f.write(xyz_coordinates.strip())
            
        cmd = f"{ORCA} calc.inp > calc.out"
        logging.info(f"Running ORCA: {cmd} (cwd={work_dir})")

        process = subprocess.run(
            cmd,
            shell=True,
            cwd=work_dir,
            capture_output=True,
            text=True,
            env=os.environ.copy(),  # 显式传递当前环境变量
        )

        if process.returncode != 0:
            raise RuntimeError(f"ORCA calculation failed: {process.stderr}")

        return OrcaResult(
            output_file=work_dir / "calc.out",
            gbw_file=work_dir / "calc.gbw",
            mol_file=work_dir / "mol.xyz",
            message="ORCA calculation completed successfully."
        )
        
    except Exception as e:
        logging.error(f"Error running ORCA calculation: {e}")
        return OrcaResult(
            output_file="",
            gbw_file="",
            mol_file="",
            message=f"ORCA calculation failed: {e}"
        )


@mcp.tool()
async def retrieve_content_from_orca_output(
    query: str, orca_outpath: Path
) -> dict:
    """
    根据查询从ORCA输出文件中检索内容。
    
    Args:
        query (str): 要搜索的查询字符串。
        orca_outpath (Path): ORCA输出文件的路径。

    Returns:
        dict: 包含状态、消息和检索到的内容的字典。
    """
    try:
        # init vector store
        vector_store = InMemoryVectorStore(embeddings)

        # Load & split the document
        loader = TextLoader(orca_outpath, encoding="utf-8")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        all_splits = text_splitter.split_documents(docs)

        # Add documents to the vector store
        document_ids = vector_store.add_documents(documents=all_splits)

        # Perform similarity search
        retrieved_docs = vector_store.similarity_search_with_score(query, k=5)

        if not retrieved_docs:
            return {
                "status": "error",
                "message": "No relevant content found.",
            }
        serialized = "\n\n".join(
            (f"Score: {score}\nContent:\n{doc.page_content}")
            for doc, score in retrieved_docs
        )

        return {
            "status": "success",
            "message": "Content retrieved successfully.",
            "retrieved_content": serialized,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to retrieve content: {str(e)}",
        }


if __name__ == "__main__":
    # Get transport type from environment variable, default to SSE
    transport_type = os.getenv('MCP_TRANSPORT', 'sse')
    mcp.run(transport=transport_type)
