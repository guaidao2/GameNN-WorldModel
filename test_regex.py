import re

# Debug the exact pattern
pat = r'\\text\{([^}]*)\}'
print('Pattern:', repr(pat))

# Test input
test = r'hello\text{world}more'
print('Test:', repr(test))

m = re.search(pat, test)
print('Match:', m)
if m:
    print('Groups:', m.groups())
else:
    # Try simpler: just match backslash
    m2 = re.search(r'\\(.)', test)
    print('Simple backslash match:', m2)
    if m2:
        print('  Group:', repr(m2.group(1)))
