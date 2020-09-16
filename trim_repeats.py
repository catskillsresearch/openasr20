import os
from trim_to_repeat import trim_to_repeat
import sys

def trim_repeats(C, cfn, tfn):
    with open(cfn, 'r', encoding='utf-8') as f:
        L=f.readlines()
    print('read', cfn)
    L1=[x.strip() for x in L]
    L2=[x.split() for x in L1]
    L3=[trim_to_repeat(trim_to_repeat(x)) for x in L2]
    result='\n'.join([' '.join(x) for x in L3])
    with open(tfn, 'w', encoding='utf-8') as f:
        f.write(result)    
    print('wrote', tfn)
