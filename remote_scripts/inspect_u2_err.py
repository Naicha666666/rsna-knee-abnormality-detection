with open('/home/eason/rsna-knee/run_remote_u2_400.py') as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if 'def call_single' in line:
        for j in range(i, min(i+40, len(lines))):
            print(f'{j+1}: {lines[j].strip()}')
        break
