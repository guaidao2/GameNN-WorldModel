import re

with open('PAPER_ZH.md', 'r', encoding='utf-8') as f:
    text = f.read()

# Find the specific user formula
idx = text.find('lambda')
if idx >= 0:
    snippet = text[idx:idx+80]
    print('Snippet:', repr(snippet))
    
    # Does it contain \text?
    if '\\\\text{' in snippet:
        print('Has \\text directly')
        # Try the fix on the snippet
        fixed = re.sub(r'\\\\text\{([^}]*)\}', r'\\\\mathrm{\1}', snippet)
        print('Fixed:', repr(fixed))
    else:
        print('No \\text found directly')
    
    # Check if it's inside inline math
    m = re.search(r'(?<![$])[$]([^$]+?)[$](?![$])', text[idx-20:idx+60])
    if m:
        print('Inline math match:', repr(m.group()))
        inner = m.group(1)
        print('Inner:', repr(inner))
        inner_fixed = re.sub(r'\\\\text\{([^}]*)\}', r'\\\\mathrm{\1}', inner)
        print('Inner fixed:', repr(inner_fixed))
