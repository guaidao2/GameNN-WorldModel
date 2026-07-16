with open('PAPER_ZH.md', 'rb') as f:
    data = f.read()

# Find the position of "text{CE}"
pos = data.find(b'text{CE}')
if pos > 0:
    print(f'Found at byte {pos}')
    # Show the 5 bytes before
    print(f'Bytes before: {data[pos-5:pos].hex()}')
    
# Also find the position of "text" followed by something
pos2 = data.find(b'lambda')
while pos2 > 0 and pos2 < 5000:
    ctx = data[pos2:pos2+30]
    print(f'Bytes at {pos2}: {ctx.hex()} -> {ctx}')
    pos2 = data.find(b'lambda', pos2+1)
