def text_of_file(fn):
    with open(fn, 'r', encoding='utf-8') as f:
        return f.read()
