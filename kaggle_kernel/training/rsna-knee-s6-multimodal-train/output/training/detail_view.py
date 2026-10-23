"""Choose the actual detail view consumed by training AND inference."""
from .anatomy_quality import choose_roi


def choose_detail_roi(anatomy,native_shape,spacing,base_mm=100.,fallback_mm=130.):
    center,extent,localized,reason=choose_roi(anatomy,native_shape,spacing,base_mm)
    if extent is not None:return center,extent,localized,reason
    if localized:
        # A supported region whose uncertainty envelope reaches the field edge
        # must not silently be shrunk back to a central 100 mm crop.
        return center,None,True,reason
    # Published source masks exposed MCL ends outside 100 mm on two specimens.
    # A fixed 130 mm fallback uses no target mask or per-case ground truth.
    # Whole-field context is always retained by the separate global view.
    return (.5,.5),max(float(base_mm),float(fallback_mm)),False,{
        **reason,'detail_fallback':'unlocalized fixed 130mm field; clinical coverage not guaranteed'}
