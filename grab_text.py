def grab_text(fn):
    with open(fn, 'r', encoding='utf-8') as f:
        return f.read()
