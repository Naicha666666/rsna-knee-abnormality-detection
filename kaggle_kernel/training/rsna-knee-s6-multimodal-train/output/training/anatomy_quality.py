"""Independent joint localization and uncertainty-aware physical ROI construction."""
from pathlib import Path
import os,sys
import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import maximum_filter
from .anatomy_runtime import native_volume


def load_localizer(weights):
    vendor=Path(__file__).parent/'vendor_runtime/kneexnet';sys.path.insert(0,str(vendor.resolve()))
    config=Path(__file__).parent/'output/yolo_config';config.mkdir(parents=True,exist_ok=True)
    os.environ['YOLO_CONFIG_DIR']=str(config.resolve());os.environ['YOLO_OFFLINE']='true'
    from ultralytics import YOLO
    return YOLO(str(weights))


def load_independent_segmenter(weights,device):
    sys.path.insert(0,str((Path(__file__).parent/'vendor_runtime/kneexnet').resolve()))
    import segmentation_models_pytorch as smp
    m=smp.Unet(encoder_name='resnet34',encoder_weights=None,in_channels=3,classes=5)
    m.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    return m.to(device).eval().requires_grad_(False)


@torch.no_grad()
def independent_tissues(model,series,joint,device):
    v=native_volume(series['path']);d,h,w=v.shape
    out=np.zeros((d,4,h,w),np.uint8);support=np.zeros((d,1,h,w),np.uint8)
    if joint.get('quality',0)<=.2:return out,support
    x1,y1,x2,y2=joint['box_xyxy_normalized'];x1,x2=int(x1*w),int(np.ceil(x2*w));y1,y2=int(y1*h),int(np.ceil(y2*h))
    slices=[]
    for sl in v:
        im=cv2.normalize(np.asarray(sl),None,0,255,cv2.NORM_MINMAX).astype(np.uint8)[y1:y2,x1:x2]
        # Native sagittal columns increase anteriorly; the released KneeX
        # spatial input increases posteriorly. Keep the independently predicted
        # crop, adapt AP for the encoder, then restore output coordinates below.
        im=im[:,::-1].copy()
        slices.append(np.asarray(Image.fromarray(im).resize((256,256),Image.Resampling.BILINEAR),np.float32)/255)
    for z in range(d):
        x=torch.from_numpy(np.stack([slices[max(0,z-1)],slices[z],slices[min(d-1,z+1)]]))[None].to(device)
        p=model(x).softmax(1)[0,[3,1,2,4]].cpu().numpy()  # author's legend: FC,TC,PC,meniscus
        p=p[:,:,::-1].copy()
        for c in range(4):out[z,c,y1:y2,x1:x2]=np.rint(cv2.resize(p[c],(x2-x1,y2-y1))*255).astype(np.uint8)
        support[z,:,y1:y2,x1:x2]=255
    return out,support


def consensus_supervision(first_u8,second_u8,observed,q):
    """Coarse localization and pixel teaching have different acceptance rules."""
    # Neither the original filter nor the integer counts mix depth/channel
    # locations. Keep identical per-pixel arithmetic while bounding temporary
    # float/bool storage to one slice instead of the entire native MRI volume.
    target=np.empty_like(first_u8,dtype=np.uint8);support=np.empty_like(first_u8,dtype=np.uint8)
    intersection=np.zeros(first_u8.shape[1],np.int64);total=np.zeros_like(intersection)
    for z in range(len(first_u8)):
        a=first_u8[z].astype(np.float32)/255;b=second_u8[z].astype(np.float32)/255
        domain=observed[z]>0;pa=a>.8;pb=b>.8;na=a<.2;nb=b<.2
        agreement=(pa&pb)|(na&nb)
        intersection+=(pa&pb&domain).sum((1,2));total+=((pa|pb)&domain).sum((1,2))
        near=maximum_filter(pa|pb,size=(1,9,9))
        support[z]=(agreement&domain&near).astype(np.uint8)*255
        target[z]=np.rint((a+b)*127.5).astype(np.uint8)
    iou=intersection/np.maximum(total,1)
    quality=q*np.where((intersection>=10)&(iou>=.25),iou,0.)*.5
    # Negatives only near predicted tissue, rather than millions of easy dark
    # pixels or an unobserved region outside the independent crop.
    return target,support,quality.astype(np.float32),{'independent_foreground_iou':iou.tolist(),
        'agreed_positive_voxels':intersection.tolist(),'role':'weak two-model agreement; not expert masks'}


def localize_series(model,series):
    volume=native_volume(series['path']);boxes=[];scores=[]
    for sl in volume:
        x=cv2.normalize(np.asarray(sl),None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
        prediction=model.predict(cv2.cvtColor(x,cv2.COLOR_GRAY2BGR),verbose=False,save=False,conf=.2)[0]
        if len(prediction.boxes):
            score=prediction.boxes.conf.cpu().numpy();j=int(score.argmax())
            boxes.append(prediction.boxes.xyxy[j].cpu().numpy());scores.append(float(score[j]))
    if len(boxes)<max(3,len(volume)//5):return {'quality':0.,'reason':'insufficient independent joint detections'}
    boxes=np.asarray(boxes);h,w=volume.shape[-2:];boxes/=np.asarray([w,h,w,h])
    centres=(boxes[:,:2]+boxes[:,2:])/2;centre=np.median(centres,axis=0)
    distance=np.linalg.norm(centres-centre,axis=1);keep=distance<max(.05,float(np.median(distance)*3))
    boxes=boxes[keep];score=np.asarray(scores)[keep]
    spread=float(np.median(np.abs(centres[keep]-centre)))
    quality=float(np.median(score)*min(1,len(boxes)/max(1,.6*len(volume)))*np.exp(-spread*8))
    box=np.r_[np.quantile(boxes[:,:2],.1,axis=0),np.quantile(boxes[:,2:],.9,axis=0)].clip(0,1)
    return {'quality':quality,'box_xyxy_normalized':box.tolist(),'centre_xy':centre.tolist(),
            'centre_spread_fraction':spread,'slices_with_detection':len(boxes),'total_slices':len(volume),
            'source':'pitthexai KneeXNet-2.5D released SagittalKneeJoint detector',
            'scope':'independent region proposal; confidence is not anatomical accuracy'}


def check_against_joint(prob,quality,joint):
    """Reject remote tissue predictions before they influence masks or pooling."""
    q=quality.copy();agreement=[]
    if joint.get('quality',0)<=.2:return q*0,{'joint_support':'unavailable; exclude unsupported regions and pixel supervision'}
    x1,y1,x2,y2=joint['box_xyxy_normalized'];h,w=prob.shape[-2:]
    # A margin retains peripheral/damaged tissues rather than enforcing a
    # normal knee silhouette. This check rejects distant islands only.
    x1,x2=max(0,x1-.08),min(1,x2+.08);y1,y2=max(0,y1-.08),min(1,y2+.08)
    support=np.zeros((h,w),bool);support[int(y1*h):int(np.ceil(y2*h)),int(x1*w):int(np.ceil(x2*w))]=True
    for c in range(4):
        weight=prob[:,c]*(prob[:,c]>.5);mass=float(weight.sum())
        inside=float((weight*support).sum())/max(mass,1e-6);agreement.append(inside)
        if inside<.8:q[c]=0.
        else:q[c]*=joint['quality']
    return q,{'fraction_in_independent_joint_envelope':agreement}


def uncertainty_regions(regions,quality,spacing_mm):
    """Expand weak regions in millimetres; never turn this into dense truth."""
    out=regions.copy();sy,sx=spacing_mm
    for c in range(regions.shape[1]):
        if quality[c]<=0:continue
        radius=2+8*(1-float(quality[c]))
        ky=2*int(np.ceil(radius/max(sy,1e-3)))+1;kx=2*int(np.ceil(radius/max(sx,1e-3)))+1
        expanded=maximum_filter(regions[:,c],size=(1,ky,kx),mode='constant')
        out[:,c]=np.maximum(regions[:,c],expanded*.5)
    return out


def choose_roi(anatomy,native_shape,spacing,base_mm=100.):
    """Explicit reliability -> physical field of view; constant across slices."""
    h,w=native_shape;sy,sx=map(float,spacing);field=np.asarray([h*sy,w*sx])
    if anatomy is None:return (.5,.5),None,False,{'reason':'no anatomy; whole field'}
    joint=anatomy.get('joint_box');jq=float(anatomy.get('joint_quality',0))
    if joint is not None and jq>.2:
        x1,y1,x2,y2=map(float,joint);centre=((y1+y2)/2,(x1+x2)/2)
        if not all(.15<v<.85 for v in centre):return (.5,.5),None,False,{'reason':'edge proposal lacks support; preserve complete field'}
        extent=max((y2-y1)*field[0],(x2-x1)*field[1])+10+30*(1-jq)
        extent=max(base_mm,extent)
        # Quantile envelope is expanded for uncertainty, not shrunk to make
        # candidate masks look clean. Full view remains a separate branch.
        if extent>=min(field):return centre,None,True,{'reason':'uncertain envelope requires complete field','quality':jq}
        return centre,float(extent),True,{'reason':'independent joint envelope plus uncertainty margin','quality':jq}
    p=anatomy['tissues'].astype(np.float32);q=anatomy['tissue_quality']
    mass=(p*(p>.5)*q[None,:,None,None]).max(1).sum(0)
    if mass.sum()<1:return (.5,.5),None,False,{'reason':'no usable candidate; whole field'}
    y,x=np.mgrid[:mass.shape[0],:mass.shape[1]];centre=(float(((y+.5)*mass).sum()/mass.sum()/mass.shape[0]),float(((x+.5)*mass).sum()/mass.sum()/mass.shape[1]))
    if not all(.15<v<.85 for v in centre) or float(q.max())<.3:
        return (.5,.5),None,False,{'reason':'weak/edge tissue-only proposal; preserve complete field'}
    reliability=float(q.max());extent=base_mm+60*(1-reliability)
    return centre,(None if extent>=min(field) else extent),True,{'reason':'weak tissue centre; expanded physical field','quality':reliability}
