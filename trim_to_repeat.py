def trim_to_repeat(S):
    n=len(S)
    max_window = n//2
    for window in range(1,max_window+1):
        for i in range(n-window):
            left=S[i:i+window]
            right=S[i+window:i+2*window]
            if left==right:
                return S[0:i+window]
            if len(S[i]) > 20:   # Really should be language specific longest word size
                return S[0:i]    # Skip super long word
    return S
