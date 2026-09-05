from __future__ import annotations

def build_model_and_rasterizer(cfg,device):
    from crackmeanflow.adapter import CrackMeanFlowModel
    from crackmeanflow.sit import build_sit
    from crackmeanflow.conference.models import build_conference_model
    from crackmeanflow.journal import build_geocrack_imf,GeometryRasterizer
    from crackmeanflow.journal.models.sit_mask_baseline import MaskIMFSiTModel,HybridMaskIMFModel
    bb=cfg['backbone'];mcfg=cfg['model']
    if bb=='unet': return build_conference_model(mcfg).to(device),None
    if bb=='geocrack_imf':
        model=build_geocrack_imf(mcfg).to(device);rast=GeometryRasterizer(mcfg.get('max_radius',16),mcfg.get('radius_bins',8),representation=mcfg.get('representation','centerline_radius'),distance_encoding=mcfg.get('distance_encoding','linear')).to(device);return model,rast
    if bb=='sit_imf_mask': return MaskIMFSiTModel(mcfg['img_size'],mcfg['patch'],mcfg['size'],mcfg.get('background_init',-.95)).to(device),None
    if bb=='hybrid_imf_mask': return HybridMaskIMFModel(mcfg['img_size'],mcfg['patch'],mcfg.get('size','S'),mcfg.get('background_init',-.95)).to(device),None
    if bb=='sit_mf':
        sit=build_sit(mcfg['img_size'],mcfg['patch'],mcfg['size'],mcfg.get('in_ch',1),mcfg.get('cond_ch',3));return CrackMeanFlowModel(sit,T=mcfg.get('T',500),ch=mcfg.get('ch')).to(device),None
    raise RuntimeError(f'unsupported backbone={bb!r}; no silent fallback is allowed')

def build_training_components(cfg,device):
    from crackmeanflow.conference.losses import ConferenceMeanFlowLoss
    from crackmeanflow.journal import ImprovedMeanFlowGeometryLoss
    from crackmeanflow.journal.flow.improved_meanflow import ImprovedMeanFlowStateLoss
    model,rast=build_model_and_rasterizer(cfg,device);bb=cfg['backbone']
    if bb in {'unet','sit_mf'}: loss=ConferenceMeanFlowLoss(**cfg['loss'])
    elif bb=='geocrack_imf': loss=ImprovedMeanFlowGeometryLoss(**cfg['loss'],max_radius=cfg['model'].get('max_radius',16),rasterizer=rast)
    elif bb in {'sit_imf_mask','hybrid_imf_mask'}: loss=ImprovedMeanFlowStateLoss(**cfg['loss'])
    else: raise RuntimeError(f'unsupported backbone={bb!r}')
    return model,rast,loss
