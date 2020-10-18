def parse_manifest(manifest):
    data = []
    for line in manifest:
        line = json.loads(line)
        data.append(line)

    return data
