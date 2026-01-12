# PyMOL-MCP: Integrating PyMOL with Claude AI

PyMOL-MCP connects PyMOL to Claude AI through the Model Context Protocol (MCP), enabling Claude to directly interact with and control PyMOL. This powerful integration allows for conversational structural biology, molecular visualization, and analysis through natural language.



https://github.com/user-attachments/assets/687f43dc-d45e-477e-ac2b-7438e175cb36



## Features

- **Two-way communication**: Connect Claude AI to PyMOL through a socket-based server
- **Intelligent command parsing**: Natural language processing for PyMOL commands
- **Molecular visualization control**: Manipulate representations, colors, and views
- **Structural analysis**: Perform measurements, alignments, and other analyses
- **Code execution**: Run arbitrary Python code in PyMOL from Claude

## Installation Guide

### Prerequisites

- PyMOL installed on your system
- Claude for Desktop
- Python 3.10 or newer
- Git

### Step 1: Install the UV Package Manager

**On macOS:**

```bash
brew install uv
```

**On Windows:**

```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
set Path=C:\Users\[YourUsername]\.local\bin;%Path%
```

For other platforms, visit the [UV installation guide](https://docs.astral.sh/uv/getting-started/installation/).

### Step 2: Clone the Repository

```bash
git clone https://github.com/vrtejus/pymol-mcp
cd pymol-mcp
```

### Step 3: Set Up the Environment

Create and activate a Python virtual environment:

```bash
python -m venv venv
```

**On macOS/Linux:**

```bash
source venv/bin/activate
```

**On Windows:**

```bash
venv\Scripts\activate
```

### Step 4: Install Dependencies

With the virtual environment activated:

```bash
pip install mcp
```

### Step 5: Configure Claude Desktop

1. Open Claude Desktop
2. Go to Claude > Settings > Developer > Edit Config
3. This will open the `claude_desktop_config.json` file
4. Add the MCP server configuration:

```json
{
  "mcpServers": {
    "pymol": {
      "command": "[Full path to your venv python]",
      "args": ["[Full path to pymol_mcp_server.py]"]
    }
  }
}
```

For example:

```json
{
  "mcpServers": {
    "pymol": {
      "command": "/Users/username/pymol-mcp/venv/bin/python",
      "args": ["/Users/username/pymol-mcp/pymol_mcp_server.py"]
    }
  }
}
```

> **Note:** Use the actual full paths on your system. On Windows, use forward slashes (/) instead of backslashes.

### Step 6: Install the PyMOL Plugin

1. Open PyMOL
2. Go to Plugin → Plugin Manager
3. Click on "Install New Plugin" tab
4. Select "Choose file..." and navigate to the cloned repository
5. Select the `pymol-mcp-socket-plugin/__init__.py` file
6. Click "Open" and follow the prompts to install the plugin

## Usage

### Starting the Connection

1. In PyMOL:

   - Go to Plugin → PyMOL MCP Socket Plugin
   - Click "Start Listening"
   - The status should change to "Listening on port 9876"

2. In Claude Desktop:
   - You should see a hammer icon in the tools section when chatting
   - Click it to access the PyMOL tools

### Example Commands

Here are some examples of what you can ask Claude to do:

- "Load PDB 1UBQ and display it as cartoon"
- "Color the protein by secondary structure"
- "Highlight the active site residues with sticks representation"
- "Align two structures and show their differences"
- "Calculate the distance between these two residues"
- "Save this view as a high-resolution image"

## Troubleshooting

- **Connection issues**: Make sure the PyMOL plugin is listening before attempting to connect from Claude
- **Command errors**: Check the PyMOL output window for any error messages
- **Plugin not appearing**: Restart PyMOL and check that the plugin was correctly installed
- **Claude not connecting**: Verify the paths in your Claude configuration file are correct

## Limitations & Notes

- The socket connection requires both PyMOL and Claude to be running on the same machine
- Some complex operations may need to be broken down into simpler steps
- Always save your work before using experimental features
- Join our Bio-MCP Community to troubleshoot, provide feedback & improve Bio-MCPS! https://join.slack.com/t/bio-mcpslack/shared_invite/zt-31z4pho39-K5tb6sZ1hUvrFyoPmKihAA

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
