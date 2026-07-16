import re

for fname in ['PAPER_ZH.md', 'PAPER_EN.md']:
    with open(fname, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Remove all math content
    no_math = re.sub(r'\$\$[^$]*\$\$', '', text)
    no_math = re.sub(r'\$[^$]*\$', '', no_math)
    no_math = re.sub(r'```.*?```', '', no_math, flags=re.DOTALL)
    no_math = re.sub(r'`[^`]*`', '', no_math)
    
    found = False
    for i, ch in enumerate(no_math):
        if ch == '_':
            if i > 0 and no_math[i-1] == '\\':
                continue
            start = max(0, i-40)
            end = min(len(no_math), i+40)
            print(f'{fname}: bare _ at pos {i}: ...{repr(no_math[start:end])}...')
            found = True
    
    if not found:
        print(f'{fname}: no bare underscores outside math')
