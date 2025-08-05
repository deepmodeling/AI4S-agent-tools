import json

ptable = 'ptable.json'
with open(ptable, 'r') as f:
    ptable_dict = json.load(f)
elements_info = ptable_dict['elements']
density_dict = {}
for ee in elements_info:
    if ee['density'] is None or ee['symbol'] is None:
        continue
    symbol = ee['symbol']
    density = ee['density'] * 1000
    density_dict[symbol] = density

    print(f"{symbol} {density}")

with open('densities.json', 'w') as f:
    json.dump(density_dict, f)
