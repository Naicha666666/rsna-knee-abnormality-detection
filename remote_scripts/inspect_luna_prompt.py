with open('/home/eason/rsna-knee/run_remote_luna_200.py') as f:
    text = f.read()
idx1 = text.find('build_teacher_prompt')
idx2 = text.find('def call_single')
print(text[idx1:idx2])
