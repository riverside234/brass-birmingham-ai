"""Small masked temporal autoencoder, PCA and authored-summary comparisons."""
from pathlib import Path
import numpy as np
import torch
from torch import nn

from brass_ai.bank import sha
from brass_ai.learning.contracts import digest,read_json,require
from brass_ai.recovery.retention import owned
from brass_ai.step5.league import write_json
from .traces import MAX_TURNS,FIELDS,schema,arrays,from_collection
from .contracts import identity

VERSION='behavior-descriptor/1.0.0'


class TemporalAE(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.GRU(len(FIELDS),32,batch_first=True);self.latent=nn.Linear(32,2)
        self.decoder=nn.GRU(4,32,batch_first=True);self.output=nn.Linear(32,len(FIELDS))

    def encode(self,x,mask):
        lengths=mask.sum(1).cpu();require(bool((lengths>0).all()),'Empty temporal sequence')
        packed=nn.utils.rnn.pack_padded_sequence(x,lengths,batch_first=True,enforce_sorted=False)
        _,h=self.encoder(packed);return self.latent(h[0])

    def forward(self,x,mask):
        z=self.encode(x,mask);position=torch.arange(MAX_TURNS,device=x.device,dtype=x.dtype)/MAX_TURNS
        time=torch.stack((position,position.square()),-1)[None].expand(len(x),-1,-1)
        h,_=self.decoder(torch.cat((z[:,None].expand(-1,MAX_TURNS,-1),time),-1))
        return z,self.output(h)


def fit_arrays(records,*,epochs=30,seed=17051,device='cpu'):
    require(records and all(r['metadata']['split']=='train' for r in records),'Descriptor fitting excludes calibration/test traces')
    require(len({r['metadata']['setup_group'] for r in records})>=2,'Fit on at least two discovery setup groups')
    require(1<=epochs<=100,'Bounded descriptor fitting required')
    x,mask=arrays(records);valid=x[mask];mean=valid.mean(0);scale=np.maximum(valid.std(0),1e-3)
    norm=np.where(mask[...,None],(x-mean)/scale,0).astype(np.float32)
    torch.manual_seed(seed);model=TemporalAE().to(device);optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    data=torch.as_tensor(norm,device=device);m=torch.as_tensor(mask,device=device)
    losses=[]
    for epoch in range(epochs):
        total=0.;count=0
        for ix in torch.randperm(len(data)).split(64):
            ix=ix.to(device);optimizer.zero_grad(set_to_none=True);_,pred=model(data[ix],m[ix])
            loss=((pred-data[ix]).square()*m[ix,:,None]).sum()/(m[ix].sum()*len(FIELDS))
            require(bool(torch.isfinite(loss)),'Nonfinite descriptor loss');loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
            total+=float(loss.detach())*len(ix);count+=len(ix)
        losses.append(total/count)
    model.eval()
    with torch.inference_mode():z=model.encode(data,m).cpu().numpy()
    latent_mean=z.mean(0);latent_scale=np.maximum(z.std(0),1e-6)
    flat=norm.reshape(len(norm),-1);pca_mean=flat.mean(0);_,_,v=np.linalg.svd(flat-pca_mean,full_matrices=False)
    basis=v[:2];require(basis.shape[0]==2,'At least two independent trajectories required')
    code=(flat-pca_mean)@basis.T;reconstructed=code@basis+pca_mean
    pca_error=float(np.square(reconstructed-flat).reshape(norm.shape)[mask].mean())
    payload=dict(format=VERSION,contract=identity(),schema=schema(),weights=model.cpu().state_dict(),mean=torch.tensor(mean),scale=torch.tensor(scale),
        latent_mean=torch.tensor(latent_mean),latent_scale=torch.tensor(latent_scale),
        pca_mean=torch.tensor(pca_mean),pca_basis=torch.tensor(basis),seed=seed)
    report=dict(epochs=epochs,loss=losses,pca_reconstruction_mse=pca_error,latent_std=z.std(0).tolist(),
        collapsed=bool((z.std(0)<1e-5).any()),parameters=sum(p.numel() for p in model.parameters()),
        traces=len(records),setups=len({r['metadata']['setup_group'] for r in records}))
    return payload,report


def transform(payload,records,*,method='ae',device='cpu'):
    require(payload['format']==VERSION and payload['contract']==identity() and payload['schema']==schema(),'Descriptor schema/source changed')
    x,mask=arrays(records);mean=payload['mean'].numpy();scale=payload['scale'].numpy()
    norm=np.where(mask[...,None],(x-mean)/scale,0).astype(np.float32)
    if method=='pca':return (norm.reshape(len(norm),-1)-payload['pca_mean'].numpy())@payload['pca_basis'].numpy().T
    if method=='authored':
        # Fixed interpretable baseline, separate from the learned coordinates.
        fields=['action_network','action_develop','action_sell','action_loan','build_Pottery','merchant_beer_used']
        selected=x[:,:,[FIELDS.index(f) for f in fields]]
        return (selected*mask[...,None]).sum(1)/mask.sum(1)[:,None]
    require(method=='ae','Unknown descriptor comparison')
    with torch.random.fork_rng(devices=[]):model=TemporalAE()
    model.load_state_dict(payload['weights'],strict=True);model=model.to(device).eval();out=[]
    with torch.inference_mode():
        for start in range(0,len(norm),64):out.append(model.encode(torch.tensor(norm[start:start+64],device=device),torch.tensor(mask[start:start+64],device=device)).cpu().numpy())
    return (np.concatenate(out)-payload['latent_mean'].numpy())/payload['latent_scale'].numpy()


def geometry(z,records):
    groups={}
    for point,r in zip(z,records):groups.setdefault(r['metadata']['candidate_id'],[]).append(point)
    centroids=np.asarray([np.mean(v,0) for v in groups.values()]);within=float(np.mean([np.square(np.asarray(v)-np.mean(v,0)).sum(-1).mean() for v in groups.values()]))
    between=float(np.var(centroids,axis=0).sum()) if len(centroids)>1 else 0.
    quality=np.asarray([r['metadata']['quality']['win_share'] for r in records]);correlations=[]
    for j in range(z.shape[1]):
        correlations.append(float(np.corrcoef(z[:,j],quality)[0,1]) if z[:,j].std()>1e-8 and quality.std()>1e-8 else 0.)
    def explained(field):
        labels=sorted({r['metadata'][field] for r in records});overall=z.mean(0)
        numerator=0.
        for value in labels:
            ix=[i for i,r in enumerate(records) if r['metadata'][field]==value]
            numerator+=len(ix)*float(np.square(z[ix].mean(0)-overall).sum())
        return numerator/max(float(np.square(z-overall).sum()),1e-8)
    neighbors=[]
    for i in range(min(len(z),16)):
        distances=np.square(z-z[i]).sum(-1);distances[i]=np.inf
        j=int(np.argmin(distances));neighbors.append(dict(trace=records[i]['metadata']['trace_id'],nearest=records[j]['metadata']['trace_id']))
    return dict(controllers=len(groups),within_variance=within,between_variance=between,repeatability_ratio=between/max(within,1e-8),
        win_correlations=correlations,seat_explained_fraction=explained('seat'),setup_explained_fraction=explained('setup_group'),
        nearest_neighbors=neighbors,scope='Descriptor repeatability/confounding diagnostic, not release style recognition')


def fit(root,collections,output,*,epochs=30,seed=17051,device='cpu'):
    root,out=owned(root,output);require(not out.exists(),'Use a new immutable descriptor directory')
    records=[];sources=[]
    for directory in collections:
        directory=Path(directory) if Path(directory).is_absolute() else root/directory
        manifest,part=from_collection(directory);require(all(r['metadata']['split']=='train' for r in part),'Held-out descriptor fit collection')
        sources.append(dict(path=directory.relative_to(root).as_posix(),manifest_sha256=sha(directory/'manifest.json'),id=manifest['id']))
        records.extend(part)
    require(len({r['metadata']['trace_id'] for r in records})==len(records),'Duplicate discovery traces')
    require(len({r['metadata']['model_id'] for r in records})==1 and len({r['metadata']['pool_id'] for r in records})==1,
        'Fit comparisons on one frozen model/opponent pool')
    payload,report=fit_arrays(records,epochs=epochs,seed=seed,device=device)
    report['comparisons']={method:geometry(transform(payload,records,method=method,device=device),records) for method in ('ae','pca','authored')}
    out.mkdir(parents=True);torch.save(payload,out/'encoder.pt')
    manifest=dict(format=VERSION,contract=identity(),schema=schema(),weights=dict(path='encoder.pt',sha256=sha(out/'encoder.pt')),sources=sources,
        training_groups=sorted({r['metadata']['setup_group'] for r in records}),trace_ids=sorted(r['metadata']['trace_id'] for r in records),report=report)
    manifest['id']=digest(manifest);write_json(out/'manifest.json',manifest)
    write_json(out/'pins.json',dict(paths=[(out/'encoder.pt').relative_to(root).as_posix()]+[s['path']+'/manifest.json' for s in sources]))
    return manifest


def load(directory):
    from brass_ai.personality.contracts import local_file
    directory=Path(directory);manifest=read_json(directory/'manifest.json')
    require(manifest['format']==VERSION and manifest['contract']==identity() and manifest['id']==digest({k:v for k,v in manifest.items() if k!='id'}),'Descriptor manifest/source changed')
    path=local_file(directory,manifest['weights']['path']);require(sha(path)==manifest['weights']['sha256'],'Descriptor weights changed')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    require(payload['schema']==schema() and payload['contract']==identity(),'Descriptor features/source changed')
    return manifest,payload
