# **AI4S-agent-tools**

An open project by the DeepModeling community - Building intelligent tools for scientific research.

🌐 **[View Tool Showcase](https://deepmodeling.github.io/AI4S-agent-tools/)** | 
🤝 **[Contribute](CONTRIBUTING.md)** |


## 🎯 Mission

We're building a comprehensive "scientific capability library" - agent-ready tools that cover the full spectrum of AI for Science tasks:

### 🔬 Current Tools

- **Materials Science** 
  - [DPACalculator](servers/DPACalculator/) - Deep learning atomistic simulations with universal potentials
  - [Thermoelectric](servers/thermoelectric/) - Materials screening with CALYPSO
  - [ABACUS-tools](servers/ABACUS-tools/) - First-principles calculations with ABACUS

- **Chemistry**
  - [PubChem](servers/pubchem/) - Compound data retrieval and structure download
  - [Catalysis](servers/catalysis/) - Reaction calculations with ADSEC workflow

- **Research Tools**
  - [Paper Search](servers/Paper_Search/) - ArXiv literature search and parsing
  - [DeepMD Docs RAG](servers/deepmd_docs_rag/) - Documentation knowledge base

### 🚀 Coming Soon

- 📊 Spectral analysis (XRD, NMR, Raman)
- 🧬 Protein structure prediction
- 🔭 3D molecular visualization
- 📈 Experimental design optimization
- 🧫 Multi-objective Bayesian optimization

## 💻 Quick Start

### Use a Tool

```bash
# Install dependencies
cd servers/<tool-name>
uv sync

# Run the server
python server.py --port 50001
```

### Add Your Tool

```bash
# Copy template
cp -r servers/_example servers/my_tool

# Edit and test
cd servers/my_tool
# ... edit server.py ...
uv sync
python server.py --port 50002
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the complete guide.

## 🏗️ Architecture

Each tool is an independent MCP (Model Context Protocol) server that can be:
- Run standalone for development
- Integrated with AI agents (Claude, GPT, etc.)
- Composed into complex workflows

## 🤝 Join Us

We welcome contributions from:
- 🧑‍🔬 Domain scientists with computational needs
- 💻 Developers interested in scientific computing
- 🤖 AI researchers building science agents
- 📚 Anyone passionate about open science

### 💬 Community

Join our WeChat community group to discuss ideas and collaborate:

<div align="center">
  <img src="data/image.png" alt="WeChat Community Group" width="200">
</div>

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with ❤️ by the [DeepModeling](https://github.com/deepmodeling) community.




