# HEA_structure_prediction MCP Server
High entropy alloy structure prediction tool
## Key Features
input : a chemical formula of a type of HEA, for example 'TiCrFeNiCuAl'
output : predict if a solid-solution system can be formed, and if so, predict the possible crystal structure. Example 'Yes, TiCrFeNiCuAl can form SS and its crystal structure is BCC+FCC

## Usage

```bash
# Install dependencies
uv sync

# Run server
python src/server.py --port 50001

# Run with custom host
python src/server.py --host localhost --port 50001

# Enable debug logging
python src/server.py --port 50001 --log-level DEBUG

## Structure

```
HEA_predict/
+---data   # data used during prediction process
|       binary_Hmix.csv
+---models # Predtrained RF models used in the prediction process
|       feats_scaler_struc.pkl
|       feats_scaler_type.pkl
|       Structure_predict_model.pkl
|       Type_predict_model.pkl
\---src
    \   server.py    # main server
```