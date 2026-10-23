import json
d = json.load(open('/home/eason/rsna-knee/u2_radimed_400_cache.json'))
errs = {k: v for k, v in d.items() if 'error' in v}
print('Total in cache:', len(d), '| Errs:', len(errs))
for k, v in list(errs.items())[:3]:
    print(k, v.get('error'))
