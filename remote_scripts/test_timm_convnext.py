import timm
print('timm version:', timm.__version__)
m = 'convnextv2_base.fcmae_ft_in22k_in1k_384'
print('is in timm:', m in timm.list_models())
try:
    bb = timm.create_model(m, pretrained=False, num_classes=0, in_chans=3)
    print('create_model success:', type(bb))
except Exception as e:
    print('create_model failed:', e)
