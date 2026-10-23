import json
d = json.load(open('/home/eason/rsna-knee/u2_radimed_400_cache.json'))
print('Total cached:', len(d), '/ 400')
