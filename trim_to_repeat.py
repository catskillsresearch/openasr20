def trim_to_repeat(S):
    n=len(S)
    for i in range(n-1):
        if S[i]==S[i+1]:
            return S[0:i+1]
    return S
