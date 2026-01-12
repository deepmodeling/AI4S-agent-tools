# Perovskite Solar Cell Data Analysis MCP Server

This is an MCP (Model Context Protocol) server for perovskite solar cell data analysis and visualization. The server provides interactive plotting tools for analyzing perovskite solar cell research trends and performance metrics.

## Key Features

1. **Interactive Data Visualization**: Generate interactive Plotly charts for perovskite solar cell research data
2. **Structure Analysis**: Analyze development trends of different perovskite solar cell structures (n-i-p, p-i-n, tandem)
3. **PCE Performance Tracking**: Track Power Conversion Efficiency (PCE) trends over time with structure-based classification
4. **Intelligent Data Processing**: Automatic structure classification and data validation
5. **Two Output Formats**: Support for both Plotly code generation and direct image rendering

## Usage

```bash
# Install dependencies
uv sync

# Run server
uv run python server.py

# Run with custom configuration
uv run python server.py --host localhost --port 50010

# Enable debug logging
uv run python server.py --log-level DEBUG
```

## Available Tools

### 1. Structure Development Trend Analysis
**Function**: `plot_solar_cell_structure_vs_time`

Analyzes the development trend of perovskite solar cell structures over time, generating normalized stacked bar charts showing the percentage distribution of different structure types (n-i-p, p-i-n, tandem, other) across multiple years.

**Parameters**:
- `start_year` (int): Starting year for analysis (2000-2030)
- `end_year` (int): Ending year for analysis (2000-2030)

**Output**: Interactive Plotly stacked bar chart with detailed yearly analysis

### 2. PCE Performance Analysis
**Function**: `plot_pce_vs_time_from_excel`

Creates interactive scatter plots showing Power Conversion Efficiency (PCE) evolution over time, with color-coded structure types. Helps visualize efficiency trends and compare performance across different device structures.

**Parameters**:
- `start_date` (str): Starting date in YYYY-MM-DD format
- `end_date` (str): Ending date in YYYY-MM-DD format

**Output**: Interactive scatter plot with statistical analysis by structure type

## Data Sources

- **Excel Files**: Currently uses `20250623_crossref.xlsx` for data analysis
- **Database Integration**: Supports database queries for real-time data access
- **Structure Classification**: Intelligent pattern matching for device structure classification

## Output Methods

The server supports two output methods for generated charts:

### 1. Plotly Code Generation
Returns Plotly figure dictionaries that can be processed by Python's plotly module to generate interactive charts. This method provides maximum flexibility for custom chart modifications.

### 2. Direct Image Rendering
Returns Cloudflare-supported object storage links for direct image display in third-party MCP clients like Cherry Studio, Cline, Cursor, etc.

## Structure Classification

The server uses intelligent pattern matching to classify perovskite solar cell structures:

- **Regular n-i-p Structure**: Contains n-i-p, nip, regular, conventional patterns
- **Inverted p-i-n Structure**: Contains p-i-n, pin, inverted, inverse patterns  
- **Tandem Structure**: Contains tandem, stack, multi-junction patterns
- **Other**: Unclassified or mixed structures

## Project Structure

```
perovskite_literature_plot/
├── server.py         # Main MCP server implementation
├── metadata.json     # Tool metadata for registry
├── pyproject.toml    # Python dependencies
└── README.md         # Tool documentation
```

## Data Privacy

**Important**: This project follows a hybrid open-source model:
- **Code**: Fully open source and available for modification and distribution
- **Data**: Not open source - proprietary research data remains confidential
- **Analysis Tools**: Open for community contribution and improvement

## Dependencies

Key dependencies managed by UV:
- `fastmcp`: MCP server framework
- `pandas`: Data processing and analysis
- `plotly`: Interactive chart generation
- `numpy`: Numerical computations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your scientific analysis tools
4. Add tests and documentation
5. Submit a pull request

## License

This project is licensed under an open-source license for the codebase. Data sources remain proprietary and confidential.

## Support

For questions about the MCP server implementation or to contribute to the codebase, please open an issue in the repository. For data access inquiries, please contact the project maintainers directly.