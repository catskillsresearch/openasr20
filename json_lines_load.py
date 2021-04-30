import json

def json_lines_load(fn):
    data = []
    with open(fn, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data
