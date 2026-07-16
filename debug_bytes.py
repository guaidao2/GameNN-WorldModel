import re

with open('PAPER_ZH.md', 'rb') as f:
    raw = f.read()

# Find the user formula in raw bytes
pattern = b'lambda_{'
pos = raw.find(pattern)
while pos > 0 and pos < 50000:
    ctx = raw[pos:pos+50]
    print(f'Bytes at {pos}: {ctx}')
    if b'text' in ctx or b'mathrm' in ctx:
        print('  Has text or mathrm!')
    pos = raw.find(pattern, pos+1)
    if pos < 0:
        break
