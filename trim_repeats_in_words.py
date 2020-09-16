import sys
language, infile, outfile = sys.argv[1:]

with open(infile,'r', encoding='utf-8') as f:
    text=f.read()
print('read', infile)

old_text=text
new_text=[]
for i in range(len(old_text)-1):
    a,b=old_text[i],old_text[i+1]
    if a != b:
        new_text.append(a)
if old_text[-2]!=old_text[-1]:
    new_text.append(old_text[-1])

text=''.join(new_text)

with open(outfile,'w', encoding='utf-8') as f:
    f.write(text)
print('wrote', outfile)
