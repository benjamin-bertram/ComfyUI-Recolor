import json, torch, numpy as np, cv2
from skimage import color as skcolor

def _tlab(r,g,b):
    t=np.array([[[r/255.,g/255.,b/255.]]])
    return skcolor.rgb2lab(t.astype(np.float64))[0,0]

def _fin(lab):
    lab[:,:,0]=np.clip(lab[:,:,0],0,100)
    lab[:,:,1]=np.clip(lab[:,:,1],-128,127)
    lab[:,:,2]=np.clip(lab[:,:,2],-128,127)
    return np.clip(skcolor.lab2rgb(lab),0,1).astype(np.float32)

def _mask(mt,shape,feather):
    m=mt[0].cpu().numpy() if mt.dim()==3 else mt.cpu().numpy()
    if m.shape[:2]!=shape[:2]: m=cv2.resize(m,(shape[1],shape[0]),interpolation=cv2.INTER_LINEAR)
    if feather>0: m=cv2.GaussianBlur(m.astype(np.float64),(0,0),feather)
    return np.clip(m,0,1)

class DirectReplaceRecolor:
    """Most precise: sets a*/b* to exact target. All detail from L*. texture_preserve: 0=flat, 0.15=subtle fabric, 0.4=heavy"""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"image":("IMAGE",),"mask":("MASK",),
            "target_r":("INT",{"default":128,"min":0,"max":255}),
            "target_g":("INT",{"default":128,"min":0,"max":255}),
            "target_b":("INT",{"default":128,"min":0,"max":255}),
            "luminance_mode":(["scale","blend","keep"],),
            "luminance_strength":("FLOAT",{"default":0.7,"min":0.0,"max":1.0,"step":0.05}),
            "texture_preserve":("FLOAT",{"default":0.15,"min":0.0,"max":1.0,"step":0.05}),
            "edge_feather":("INT",{"default":2,"min":0,"max":50})}}
    RETURN_TYPES=("IMAGE",); RETURN_NAMES=("recolored",); FUNCTION="process"; CATEGORY="AICG/Recolor"
    def process(self,image,mask,target_r,target_g,target_b,luminance_mode="scale",luminance_strength=0.7,texture_preserve=0.15,edge_feather=2):
        img=image[0].cpu().numpy(); lab=skcolor.rgb2lab(img.astype(np.float64))
        tL,ta,tb=_tlab(target_r,target_g,target_b); m=_mask(mask,img.shape,edge_feather)
        out=lab.copy(); mb=m>0.05
        if mb.sum()==0: return (image,)
        sa,sb,sL=lab[mb,1].mean(),lab[mb,2].mean(),lab[mb,0].mean()
        if texture_preserve>0:
            na=ta+(lab[:,:,1]-sa)*texture_preserve; nb=tb+(lab[:,:,2]-sb)*texture_preserve
        else:
            na=np.full_like(lab[:,:,1],ta); nb=np.full_like(lab[:,:,2],tb)
        out[:,:,1]=lab[:,:,1]*(1-m)+na*m; out[:,:,2]=lab[:,:,2]*(1-m)+nb*m
        if luminance_mode=="scale" and sL>0.5:
            eff=1.0+(tL/sL-1.0)*luminance_strength; out[:,:,0]=lab[:,:,0]*(1-m)+(lab[:,:,0]*eff)*m
        elif luminance_mode=="blend":
            out[:,:,0]=lab[:,:,0]+(tL-sL)*luminance_strength*m
        return (torch.from_numpy(_fin(out)).unsqueeze(0),)

class StatisticalTransferRecolor:
    """Reinhard-style: normalize+remap. target_spread: 0=flat, 0.25=subtle, 1.0=full variation"""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"image":("IMAGE",),"mask":("MASK",),
            "target_r":("INT",{"default":128,"min":0,"max":255}),
            "target_g":("INT",{"default":128,"min":0,"max":255}),
            "target_b":("INT",{"default":128,"min":0,"max":255}),
            "luminance_strength":("FLOAT",{"default":0.7,"min":0.0,"max":1.0,"step":0.05}),
            "target_spread":("FLOAT",{"default":0.25,"min":0.0,"max":1.0,"step":0.05}),
            "edge_feather":("INT",{"default":2,"min":0,"max":50})}}
    RETURN_TYPES=("IMAGE",); RETURN_NAMES=("recolored",); FUNCTION="process"; CATEGORY="AICG/Recolor"
    def process(self,image,mask,target_r,target_g,target_b,luminance_strength=0.7,target_spread=0.25,edge_feather=2):
        img=image[0].cpu().numpy(); lab=skcolor.rgb2lab(img.astype(np.float64))
        tL,ta,tb=_tlab(target_r,target_g,target_b); m=_mask(mask,img.shape,edge_feather)
        out=lab.copy(); mb=m>0.05
        if mb.sum()==0: return (image,)
        sLm,sLs=lab[mb,0].mean(),max(lab[mb,0].std(),0.01)
        sam,sas=lab[mb,1].mean(),max(lab[mb,1].std(),0.01)
        sbm,sbs=lab[mb,2].mean(),max(lab[mb,2].std(),0.01)
        na=ta+((lab[:,:,1]-sam)/sas)*sas*target_spread
        nb=tb+((lab[:,:,2]-sbm)/sbs)*sbs*target_spread
        out[:,:,1]=lab[:,:,1]*(1-m)+na*m; out[:,:,2]=lab[:,:,2]*(1-m)+nb*m
        if sLm>0.5:
            nL=tL+((lab[:,:,0]-sLm)/sLs)*sLs
            bl=lab[:,:,0]*(1-luminance_strength)+nL*luminance_strength
            out[:,:,0]=lab[:,:,0]*(1-m)+bl*m
        return (torch.from_numpy(_fin(out)).unsqueeze(0),)

class PercentileMapRecolor:
    """Robust percentile mapping. target_range_pct: 0=flat, 0.25=compressed, 1.0=full"""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"image":("IMAGE",),"mask":("MASK",),
            "target_r":("INT",{"default":128,"min":0,"max":255}),
            "target_g":("INT",{"default":128,"min":0,"max":255}),
            "target_b":("INT",{"default":128,"min":0,"max":255}),
            "luminance_strength":("FLOAT",{"default":0.7,"min":0.0,"max":1.0,"step":0.05}),
            "target_range_pct":("FLOAT",{"default":0.25,"min":0.0,"max":1.0,"step":0.05}),
            "edge_feather":("INT",{"default":2,"min":0,"max":50})}}
    RETURN_TYPES=("IMAGE",); RETURN_NAMES=("recolored",); FUNCTION="process"; CATEGORY="AICG/Recolor"
    def process(self,image,mask,target_r,target_g,target_b,luminance_strength=0.7,target_range_pct=0.25,edge_feather=2):
        img=image[0].cpu().numpy(); lab=skcolor.rgb2lab(img.astype(np.float64))
        tL,ta,tb=_tlab(target_r,target_g,target_b); m=_mask(mask,img.shape,edge_feather)
        out=lab.copy(); mb=m>0.05
        if mb.sum()==0: return (image,)
        def remap(ch,tc,s=1.0):
            v=ch[mb]; pl,ph=np.percentile(v,5),np.percentile(v,95); sr=max(ph-pl,0.01)
            thr=(sr*target_range_pct)/2; n=np.clip((ch-pl)/sr,0,1)
            return ch*(1-s)+(tc-thr+n*thr*2)*s
        out[:,:,1]=lab[:,:,1]*(1-m)+remap(lab[:,:,1],ta)*m
        out[:,:,2]=lab[:,:,2]*(1-m)+remap(lab[:,:,2],tb)*m
        out[:,:,0]=lab[:,:,0]*(1-m)+remap(lab[:,:,0],tL,luminance_strength)*m
        return (torch.from_numpy(_fin(out)).unsqueeze(0),)

class MultiZoneRecolor:
    """Multi-zone with method selection. zone_config: [{"mask_index":0,"r":140,"g":137,"b":189}]"""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"image":("IMAGE",),"zone_config":("STRING",{"default":"[]","multiline":True}),
            "method":(["direct_replace","statistical","percentile"],),
            "luminance_strength":("FLOAT",{"default":0.7,"min":0.0,"max":1.0,"step":0.05}),
            "texture_preserve":("FLOAT",{"default":0.15,"min":0.0,"max":1.0,"step":0.05}),
            "edge_feather":("INT",{"default":2,"min":0,"max":50})},
            "optional":{"mask_0":("MASK",),"mask_1":("MASK",),"mask_2":("MASK",),"mask_3":("MASK",),
                "mask_4":("MASK",),"mask_5":("MASK",),"mask_6":("MASK",),"mask_7":("MASK",)}}
    RETURN_TYPES=("IMAGE",); RETURN_NAMES=("recolored",); FUNCTION="process"; CATEGORY="AICG/Recolor"
    def process(self,image,zone_config,method="direct_replace",luminance_strength=0.7,texture_preserve=0.15,edge_feather=2,
                mask_0=None,mask_1=None,mask_2=None,mask_3=None,mask_4=None,mask_5=None,mask_6=None,mask_7=None):
        zones=json.loads(zone_config); masks=[mask_0,mask_1,mask_2,mask_3,mask_4,mask_5,mask_6,mask_7]
        nodes={"direct_replace":DirectReplaceRecolor,"statistical":StatisticalTransferRecolor,"percentile":PercentileMapRecolor}
        node=nodes[method]()
        cur=image.clone()
        for z in zones:
            idx=z["mask_index"]
            if idx<len(masks) and masks[idx] is not None:
                if method=="direct_replace":
                    cur=node.process(cur,masks[idx],z["r"],z["g"],z["b"],"scale",luminance_strength,texture_preserve,edge_feather)[0]
                elif method=="statistical":
                    cur=node.process(cur,masks[idx],z["r"],z["g"],z["b"],luminance_strength,texture_preserve,edge_feather)[0]
                else:
                    cur=node.process(cur,masks[idx],z["r"],z["g"],z["b"],luminance_strength,texture_preserve,edge_feather)[0]
        return (cur,)

class BatchColorwayProcessor:
    """All colorways in one pass. colorways_json: [{"name":"JW4388","zones":[...]}]"""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"image":("IMAGE",),"colorways_json":("STRING",{"default":"[]","multiline":True}),
            "method":(["direct_replace","statistical","percentile"],),
            "luminance_strength":("FLOAT",{"default":0.7,"min":0.0,"max":1.0,"step":0.05}),
            "texture_preserve":("FLOAT",{"default":0.15,"min":0.0,"max":1.0,"step":0.05}),
            "edge_feather":("INT",{"default":2,"min":0,"max":50})},
            "optional":{"mask_0":("MASK",),"mask_1":("MASK",),"mask_2":("MASK",),"mask_3":("MASK",),
                "mask_4":("MASK",),"mask_5":("MASK",),"mask_6":("MASK",),"mask_7":("MASK",)}}
    RETURN_TYPES=("IMAGE","STRING"); RETURN_NAMES=("batch","names"); FUNCTION="process"; CATEGORY="AICG/Recolor"
    def process(self,image,colorways_json,method="direct_replace",luminance_strength=0.7,texture_preserve=0.15,edge_feather=2,
                mask_0=None,mask_1=None,mask_2=None,mask_3=None,mask_4=None,mask_5=None,mask_6=None,mask_7=None):
        cws=json.loads(colorways_json); mn=MultiZoneRecolorV2(); results=[]; names=[]
        for cw in cws:
            r=mn.process(image,json.dumps(cw.get("zones",[])),method,luminance_strength,texture_preserve,edge_feather,
                mask_0,mask_1,mask_2,mask_3,mask_4,mask_5,mask_6,mask_7)[0]
            results.append(r); names.append(cw.get("name",""))
        return (torch.cat(results,dim=0) if results else image,", ".join(names))

class AutoColorZoneSegmenter:
    """K-Means zone detection."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"image":("IMAGE",),"product_mask":("MASK",),"num_zones":("INT",{"default":3,"min":2,"max":10})}}
    RETURN_TYPES=("MASK","MASK","MASK","MASK","MASK","MASK","MASK","MASK","STRING")
    RETURN_NAMES=("z0","z1","z2","z3","z4","z5","z6","z7","info"); FUNCTION="segment"; CATEGORY="AICG/Recolor"
    def segment(self,image,product_mask,num_zones=3):
        img=(image[0].cpu().numpy()*255).astype(np.uint8); m=_mask(product_mask,img.shape,0); mb=m>0.5
        ilab=cv2.cvtColor(img,cv2.COLOR_RGB2LAB); px=ilab[mb].reshape(-1,3).astype(np.float32)
        crit=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,500,0.5)
        _,labels,_=cv2.kmeans(px,num_zones,None,crit,10,cv2.KMEANS_PP_CENTERS)
        lmap=np.full(img.shape[:2],-1,dtype=np.int32); lmap[mb]=labels.flatten()
        zones=[]; info=[]; empty=torch.zeros(1,img.shape[0],img.shape[1])
        for i in range(num_zones):
            zm=(lmap==i).astype(np.float32); k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            zm=cv2.morphologyEx(cv2.morphologyEx(zm,cv2.MORPH_CLOSE,k),cv2.MORPH_OPEN,k)
            zones.append(torch.from_numpy(zm).unsqueeze(0))
            p=img[lmap==i]
            if len(p)>0: a=p.mean(axis=0).astype(int); info.append(f"Zone {i}: RGB({a[0]},{a[1]},{a[2]}) {len(p)/mb.sum()*100:.1f}%")
        while len(zones)<8: zones.append(empty)
        return (*zones[:8],"\n".join(info))

class RGBColorInput:
    """Parse RGB string."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"rgb_string":("STRING",{"default":"128, 128, 128"}),"label":("STRING",{"default":"Main"})}}
    RETURN_TYPES=("INT","INT","INT","STRING"); RETURN_NAMES=("R","G","B","info"); FUNCTION="parse"; CATEGORY="AICG/Recolor"
    def parse(self,rgb_string,label="Main"):
        p=[int(x.strip()) for x in rgb_string.split(",")]; return (p[0],p[1],p[2],f"{label}: RGB({p[0]},{p[1]},{p[2]})")

NODE_CLASS_MAPPINGS={
    "DirectReplaceRecolor":DirectReplaceRecolor,"StatisticalTransferRecolor":StatisticalTransferRecolor,
    "PercentileMapRecolor":PercentileMapRecolor,"MultiZoneRecolor":MultiZoneRecolor,
    "BatchColorwayProcessor":BatchColorwayProcessor,"AutoColorZoneSegmenter":AutoColorZoneSegmenter,
    "RGBColorInput":RGBColorInput}
NODE_DISPLAY_NAME_MAPPINGS={
    "DirectReplaceRecolor":"🎨 Direct Replace Recolor","StatisticalTransferRecolor":"🎨 Statistical Transfer",
    "PercentileMapRecolor":"🎨 Percentile Map Recolor","MultiZoneRecolor":"🎨 Multi-Zone Recolor",
    "BatchColorwayProcessor":"⚡ Batch Colorway","AutoColorZoneSegmenter":"🔍 Auto Zone Segmenter",
    "RGBColorInput":"🎯 RGB Color Input"}
