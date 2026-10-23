with open('/home/eason/rsna-knee/run_remote_u2_400.py') as f:
    lines = f.readlines()
for i in range(110, min(145, len(lines))):
    print(f'{i+1}: {lines[i].strip()}')
