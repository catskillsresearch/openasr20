# coding: utf-8

import sys
language, infer_fn, cfn = sys.argv[1:]

# # Add error-correction based on model eval results
import json, os
from tokenizer import tokenizer
from error_correct import error_correct

stage='NIST'

with open(infer_fn, 'r', encoding='utf-8') as f:
    L=f.readlines()
print('read', infer_fn)

L1=[tokenizer(x) for x in L]

afn = f'analysis/{language}/afterburner.json'
with open(afn, 'r') as f:
    M=json.load(f)

L2=[error_correct(pred,M) for pred in L1]

result='\n'.join([' '.join(x) for x in L2])

with open(cfn, 'w', encoding='utf-8') as f:
    f.write(result)    
print('wrote', cfn)
