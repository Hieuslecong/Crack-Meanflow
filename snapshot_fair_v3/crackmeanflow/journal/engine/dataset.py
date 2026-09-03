from torch.utils.data import Dataset
from crackmeanflow.journal.geometry.targets import mask_to_geometry_state

class GeometryDataset(Dataset):
    def __init__(self,base,max_radius=16.,representation='centerline_radius',distance_encoding='linear'):
        self.base=base
        self.max_radius=float(max_radius)
        self.representation=str(representation)
        self.distance_encoding=str(distance_encoding)
    def __len__(self): return len(self.base)
    def __getitem__(self,i):
        item=self.base[i]
        g,v=mask_to_geometry_state(item['mask'][None],self.max_radius,self.representation,self.distance_encoding)
        out=dict(item); out['geometry']=g[0]; out['radius_valid']=v[0]; return out
