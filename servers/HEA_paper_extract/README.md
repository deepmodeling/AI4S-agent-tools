# HEA_extractTool
extract structural data of HEA materials from searched or given publications
## Key Features
Search_paper
input : 
      user_query, example: High entropy alloy
      search_type: title, author of all (for keywords)
      max_results: int, expected number of results
output : saved path of the downloaded pdf files

HEA_data_extract
input :
      manuscript: path to the pdf file of a paper
      out_dir: output directory, default 'results' 
output :
      if the manuscript doesn't include any HEA materials research, return"No, the author did not study the phase structure and properties of High Entropy alloys"
      otherwise, return the path to the extracted csv file

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

