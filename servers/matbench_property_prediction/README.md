# Matbench Property Prediction MCP Server
Predict matbench properties using deep potential models, including refractive index(unitless), exfoliation energy (meV/atom), the DFT Voigt-Reuss-Hill average shear moduli in GPa, the DFT Voigt-Reuss-Hill average bulk moduli in GPa, formation energy in eV as calculated by the Materials Project, the band gap as calculated by PBE DFT from the Materials Project, heat of formation of the entire 5-atom perovskite cell in eV as calculated by RPBE GGA-DFT and frequency of the highest frequency optical phonon mode peak in units of 1/cm. If user did not mention specific matbench properties please calculate all supported matbench properties.

## Usage

```bash
# Install dependencies
uv sync

# Run server
python server.py --port 50001

# Run with custom host
python server.py --host localhost --port 50001

# Enable debug logging
python server.py --port 50001 --log-level DEBUG
```

## Example
Input : 
     Can you give me the frequency of the highest frequency optical phonon mode peak of structures in xxx?
Output : 
     The frequency of the highest frequency optical phonon mode peak for the structure "O32Si16" is approximately 1184.08 cm⁻¹.