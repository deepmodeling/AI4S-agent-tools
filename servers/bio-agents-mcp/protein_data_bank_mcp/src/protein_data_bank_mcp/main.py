from protein_data_bank_mcp.server import mcp_server

if __name__ == "__main__":
    mcp_server.run(transport="sse")
