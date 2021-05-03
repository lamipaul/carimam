from tqdm import tqdm
from scipy import signal
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import umap
import hdbscan

folder = '../DATA/BONAIRE/session1_20201217to20210126/'
configs = [
{'id':'mel_BBF', 'fs': 2000},
{'id':'mel_HBF', 'fs': 16000},
{'id':'mel_BMF', 'fs': 64000},
{'id':'mel_HMF', 'fs': 256000},
{'id':'stft_HMF', 'fs': 64000},
{'id':'stft_HF', 'fs': 256000}
]

sort = True
fns = pd.Series(os.listdir(folder))
fns = fns[fns.str.endswith('_spec.npy')] #.sample(500) # change here to select a subset of files
fs = 512000
chunksize = 20 # ~10sec chunks

for c in configs[1:]:
    print('loading specs for '+c['id'])
    X, meta = [], []
    for f in tqdm(fns):
        a = np.load(folder+f, allow_pickle=True).item()[c['id']]
        for offset in np.arange(0, a.shape[1]-chunksize, chunksize): 
            temp = np.log10(a[:,offset:offset+chunksize])
            if sort:
                temp = np.vstack([np.flip(np.sort(l)) for l in temp])
            X.append(temp[:, [1, 3, 5]]) # choice of quantiles ! (nth highest bin over chunksize)
            meta.append({'fn':f, 'offset':offset})
    X = np.array(X)
    meta = pd.DataFrame().from_dict(meta)
    X = X.reshape(len(X), -1)
    project = umap.UMAP()
    embed = project.fit_transform(X)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=10).fit(embed) # au depart 50 et 3
    meta['cluster'] = clusterer.labels_
    print('clusters for '+c['id']+' (fn : n files found in clusters, offset : n samples in cluster)')
    print(meta.groupby('cluster').agg({'fn':'nunique', 'offset':'count'}))

    plt.scatter(embed[:,0], embed[:,1], c=clusterer.labels_ , cmap="Paired", s=1)
    plt.show()
    plt.savefig('scatter_'+c['id'])
    

    sos = signal.butter(3, c['fs']/fs, 'lp', output='sos')

    for cluster, grp in meta.groupby('cluster'):
        if len(grp) > 300:
            continue
        os.system('mkdir -p pngs/'+c['id']+'/'+str(cluster))
        for r in tqdm(grp.itertuples(), desc=str(cluster), total=len(grp), leave=False):
            sig, fs = sf.read('../DATA/BONAIRE/session1_20201217to20210126/'+r.fn[:-9]+'.WAV', start=int(r.offset*fs), stop=int((r.offset+5)*fs))
            if len(sig) < fs*5:
                continue
            sig = signal.sosfiltfilt(sos, sig)[::fs//c['fs']]
            plt.specgram(sig, NFFT=1024, noverlap=512, Fs=c['fs'])
            plt.savefig('pngs/'+c['id']+'/'+str(cluster)+'/'+r.fn[:-9]+'_'+str(r.offset))
            plt.close()
