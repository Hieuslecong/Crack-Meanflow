from .mlt_unet import UNet
from crackmeanflow.adapter import CrackMeanFlowModel

def build_conference_model(model_cfg):
    unet=UNet(T=model_cfg.get('T',500),ch=model_cfg['ch'],ch_mult=list(model_cfg['ch_mult']),attn=list(model_cfg.get('attn',[])),num_res_blocks=model_cfg['num_res_blocks'],dropout=model_cfg.get('dropout',0.1))
    return CrackMeanFlowModel(unet,T=model_cfg.get('T',500),ch=model_cfg['ch'])
