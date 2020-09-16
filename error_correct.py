def error_correct(S, M):
    return [M[x] if x in M else x for x in S]
