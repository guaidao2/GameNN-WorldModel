import re

for fname in ['PAPER_ZH.md', 'PAPER_EN.md']:
    with open(fname, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # In inline math $...$, replace \text{XXX} with \mathrm{XXX} 
    # for common abbreviations: CE, MSE, BCE, strategy, action, success, failure, etc.
    # Strategy: find \text{ inside $...$ and replace with \mathrm{
    
    def fix_inline_math(m):
        content = m.group(1)
        # Replace \text{...} with \mathrm{...} inside inline math
        content = re.sub(r'\\text\{([^}]*)\}', r'\\mathrm{\1}', content)
        return '$' + content + '$'
    
    # Match inline math $...$ but not display math $$...$$
    text = re.sub(r'(?<!\$)\$([^$]+?)\$(?!\$)', fix_inline_math, text)
    
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'{fname}: fixed')
"
