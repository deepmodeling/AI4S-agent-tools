# ASKCOS MCP Server

This is an MCP server that interfaces with ASKCOS for computer-aided synthesis planning and retrosynthesis analysis.

## Key Features

1. **Retrosynthesis Planning**: Analyze synthetic routes for target molecules
2. **Synthesis Route Analysis**: Evaluate and compare different synthetic pathways
3. **Reaction Prediction**: Predict chemical reactions and their outcomes
4. **ASKCOS API Integration**: Seamless connection to ASKCOS backend services

## Prerequisites

Before using this tool, you need to have an ASKCOS instance running. Please refer to the [ASKCOS documentation](https://askcos-docs.mit.edu/) for deployment instructions.

## Usage

```bash
# Install dependencies
uv sync

# Set ASKCOS API URL (required)
export ASKCOS_API_URL="http://your-askcos-instance:9100"

# Run server
python server.py --port 50001

# Run with custom host
python server.py --host localhost --port 50001

# Enable debug logging
python server.py --port 50001 --log-level DEBUG
```

## Configuration

1. Set the `ASKCOS_API_URL` environment variable to point to your ASKCOS instance
2. Ensure your ASKCOS instance is running and accessible
3. Install the required dependencies using `uv sync`
4. Test the connection to your ASKCOS API
5. Run the MCP server

## Structure

```
ASKCOS/
├── server.py         # ASKCOS MCP server implementation
├── metadata.json     # Tool metadata for registry
├── pyproject.toml    # Python dependencies
├── README.md         # Tool documentation
└── ASKCOS.md         # ASKCOS integration guide
```

## Available Tools

### retrosynthesis_planning
Performs retrosynthesis planning for a given molecule SMILES string.

**Parameters:**
- `smiles` (string): The SMILES representation of the target molecule
- `max_depth` (integer, optional): Maximum depth for the synthesis tree

**Returns:**
- Synthesis plan with reaction steps and intermediates

### synthesis_route_analysis
Analyzes and evaluates different synthetic routes for a target molecule.

**Parameters:**
- `smiles` (string): The SMILES representation of the target molecule
- `routes` (array, optional): Specific routes to analyze

**Returns:**
- Analysis results with route rankings and feasibility scores

### reaction_prediction
Predicts possible reactions for given reactants.

**Parameters:**
- `reactants` (array): List of reactant SMILES strings
- `conditions` (object, optional): Reaction conditions

**Returns:**
- Predicted reaction products and pathways

## Environment Variables

- `ASKCOS_API_URL`: URL of your ASKCOS instance (required)
- `ASKCOS_USERNAME`: Username for ASKCOS authentication (optional)
- `ASKCOS_PASSWORD`: Password for ASKCOS authentication (optional)

## Troubleshooting

- Ensure your ASKCOS instance is running and accessible
- Check that the API URL is correct and includes the port (usually 9100)
- Verify that your ASKCOS instance has the required models loaded
- Check the server logs for detailed error messages

## References

- [ASKCOS Documentation](https://askcos-docs.mit.edu/)
- [ASKCOS GitHub Repository](https://github.com/ASKCOS/askcos-core)
- [AI4S-agent-tools Documentation](../../README.md)