import sys
sys.path.insert(0, '/home/eason/rsna-knee/src')
sys.path.insert(0, '/home/eason/rsna-knee/pylib-cu')
import tv_stub; tv_stub.install()
import timm
print([m for m in timm.list_models('*convnextv2*') if '384' in m or 'base' in m][:10])
