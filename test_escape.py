print('Testing single escape:')
a = '\x5c'  # backslash via hex escape
print(f'a = {repr(a)} len={len(a)} ord={hex(ord(a[0]))}')

b = '\x5c' + 't'
print(f'b = {repr(b)} len={len(b)} chars={[hex(ord(c)) for c in b]}')

# What I wrote in the script: '\\text{CE}'
# This means: backslash-escape + backslash-escape + t + e + x + t + ...
# In Python: \\ is one backslash, then t is literal t
c = '\\' + '\\' + 't' + 'ext{CE}'
print(f'c = {repr(c)} len={len(c)} chars={[hex(ord(c0)) for c0 in c]}')

# The correct way
d = '\x5ctext{CE}'  # backslash + text{CE}
print(f'd = {repr(d)} len={len(d)}')
print(f'd[0] == chr(92): {d[0] == chr(92)}')
