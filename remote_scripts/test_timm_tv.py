import sys
sys.path.insert(0, '/home/eason/rsna-knee/src')
sys.path.insert(0, '/home/eason/rsna-knee/pylib-cu')
import tv_stub
tv_stub.install()
import timm
print('timm version:', timm.__version__)
m = 'convnextv2_base.fcmae_ft_in22k_in1k_384'
print('is in timm:', m in timm.list_models())
bb = timm.create_model(m, pretrained=False, num_classes=0, in_chans=3)
print('created:', type(bb), 'num_features:', bb.num_features)
