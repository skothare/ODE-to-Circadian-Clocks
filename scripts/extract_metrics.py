import re

try:
    with open('estimation_output.txt', 'r', encoding='utf-16') as f:
        content = f.read()
except UnicodeError:
    with open('estimation_output.txt', 'r', encoding='utf-8') as f:
        content = f.read()

lines = content.split('\n')
metrics = []
for line in lines:
    if 'Phase Shift' in line:
        # Clean up garbled characters if any
        line = line.replace('R▓', 'R^2')
        metrics.append(line.strip())

with open('metrics.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(metrics))

print("Extracted metrics:")
print('\n'.join(metrics))
