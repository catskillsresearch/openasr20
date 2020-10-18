import json

def json_lines_save(L, fn):
    with open(fn, 'w', encoding='utf-8') as f:
        for line in L:
            json.dump(line, f)
            f.write('\n')
    print('saved', fn)
