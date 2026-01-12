"""
Collect data from an ABACUS job and dump to json format compatible with whole dataset
"""
import json
from abacustest.lib_prepare.abacus import ReadInput

input_params = ReadInput("INPUT")

stru_file_content = []
with open("STRU") as fin:
    for lines in fin:
        lines = lines.strip()
        if lines != '':
            stru_file_content.append(lines)

data = {
    'description': None,
    'input': input_params,
    'stru': stru_file_content,
    'pp': None,
    'orb': None
}

if 'suffix' in input_params:
    suffix = input_params['suffix']
else:
    suffix = 'ABACUS'

with open(f"{suffix}-data.json", "w") as fin:
    json.dump(data, fin, indent=4)

