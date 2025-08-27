import argparse
import pandas as pd
import arxiv
import os
import json
import re
import pathlib
import requests
from dependencies.HEA_extractor import chat_chain_get_msg
from mcp.server.fastmcp import FastMCP
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
            port = 50002
            host = '0.0.0.0'
            log_level = 'INFO'
        args = Args()
    return args
args = parse_args()
mcp = FastMCP("example",host=args.host,port=args.port)

PAPER_DIR ='paper'
@mcp.tool()
def search_paper(user_query: str,search_type:str, max_results: int = 5) -> dict:
    """
    Search for papers on arXiv by title, author or keywords,
    download the original publications and save basic information.
    read the info file to give information detials
    Args:
        user_query: string
        query_type: 'title', 'author', or 'all'
        max_results: Maximum number of results to retrieve (default: 5)
    Returns:
        path: The directory where papers and info are saved.
    """
    # Use arxiv to find the papers 
    client = arxiv.Client()
    # Search for papers where the title/author/all matches the input
    if  search_type == 'title':
        search_query = f'ti:"{user_query}"'
    elif search_type == 'author':
        search_query = f'au:"{user_query}"'
    elif search_type == 'all':
        search_query = f'all:"{user_query}"'
    else:
        raise ValueError("search_type must be 'title', 'author', or 'all'")
    
    search = arxiv.Search(
               query=search_query,  
               max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )  
    papers = client.results(search)

    # Directory management (optional, like original function)
    path = os.path.join(PAPER_DIR, user_query.lower().replace(" ", "_"))
    os.makedirs(path, exist_ok=True)
    info_path = os.path.join(path, "papers_info.json")
    # Try to load existing papers info
    try:
        with open(info_path, "r") as json_file:
            papers_info = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        papers_info = {}
    # Process and download each paper and add to papers_info  
    paper_ids = []
    for paper in papers:
        paper_id = paper.get_short_id()
        paper_ids.append(paper_id)
        paper_info = {
            'title': paper.title,
            'authors': [author.name for author in paper.authors],
            'summary': paper.summary,
            'pdf_url': paper.pdf_url,
            'published': str(paper.published.date())
        }
        safe_title = paper.title[:50]
        pdf_filename = re.sub(r'[-\\/*?:"<>| ]', "_", safe_title) + ".pdf"
        file_path = os.path.join(path,pdf_filename)
        print(f"Downloading {paper.pdf_url} -> {file_path}")
        try:
            pdf_resp = requests.get(paper.pdf_url)
            if pdf_resp.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(pdf_resp.content)
                print(f"Saved: {file_path}")
                paper_info['pdf_path']=file_path
            else:
                print(f"Failed to download {paper.pdf_url}: {pdf_resp.status_code}")
                paper_info['pdf_path']=None
        except Exception as e:
                print(f"Error downloading {paper.pdf_url}: {e}")
                paper_info['pdf_path']=None
        papers_info[paper_id] = paper_info

    # Save updated papers_info to json file
    with open(info_path, "w") as json_file:
        json.dump(papers_info, json_file, indent=2, ensure_ascii=False)
    print(f"Results and PDFs are saved in: {path}")
    return papers_info

OUT_DIR_temp = 'results'
@mcp.tool()
def HEA_data_extract(manuscript:str, out_dir = OUT_DIR_temp)->str:
    '''
    Args: 
       manucript: .pdf file path of the manuscript to be analysed
       out_dir: output path
    
    extract structural data of High Entropy Alloy from a manuscript
    including: name,composition,detailed phase structure,mechanical properties
    save results as csv file
    return the csv file path'''
    return chat_chain_get_msg(
    manuscript=manuscript,
    support_si_dir=None,  # 如有补充材料可指定目录
    out_dir=out_dir,
    del_old=True,
    debug=False,
    model_json=None,      # 若有已生成的json，可加速流程
    verbose=True,
    warm_start=False,
    warm_prompt=None
)


if __name__ == "__main__":
    # Get transport type from environment variable, default to SSE
    transport_type = os.getenv('MCP_TRANSPORT', 'sse')
    mcp.run(transport=transport_type)