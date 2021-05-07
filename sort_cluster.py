from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import umap
import hdbscan


folder = '../DATA/BONAIRE/session1_20201217to20210126/' # path to a given recording station folder
# list of spectrogram configs to be clustered
configs = [
    {'id':'mel_BBF', 'fs': 2000},
    {'id':'mel_HBF', 'fs': 16000},
    {'id':'mel_BMF', 'fs': 64000},
    {'id':'mel_HMF', 'fs': 256000},
    {'id':'stft_HMF', 'fs': 64000},
    {'id':'stft_HF', 'fs': 256000}
]

sort = True # whether we sort each frequency bins by descending order (used to build an energy distribution image and eliminate time dependent features)

#get filenames list, filter _spec.npy files only (possibly sample a subset randomly for testing)
fns = pd.Series(os.listdir(folder))
fns = fns[fns.str.endswith('_spec.npy')] #.sample(500)

fs = 512000 # TODO adapt to each recording station (some run at 256kHz)
chunksize = 20 # size (in time bins) of chunks of spectrogram to cluster independantly

# for each configuration, we load spectrograms, project features, cluster, and plot spectrograms in pngs
for c in configs:
    # arrays X and meta will hold features and metadata for each samples to be projected / clustered
    X, meta = [], []
    for f in tqdm(fns, desc='loading specs for '+c['id'], leave=False):
        # load the spectrogram from the .npy file
        spectro = np.load(folder+f, allow_pickle=True).item()[c['id']]
        
        # cut the spectrogram in chunks time wise
        for offset in np.arange(0, spectro.shape[1]-chunksize, chunksize): 
            # get logarithmic magnitude
            temp = np.log10(spectroDict[:,offset:offset+chunksize])
            if sort:
                # if sort is True, we sort each frequency bin in descending order
                temp = np.flip(np.sort(temp, axis=1), axis=1)
                # we can then select a subset of bins (similar to quantiles)
                X.append(temp[:, [1, 3, 5]]) 
            else : 
                # else, we use the whole spectrogram as input features for projection
                X.append(temp)
            # save filename and offset to retrieve the sample later on
            meta.append({'fn':f, 'offset':offset})
    
    # rearange X and meta arrays for easier later use
    X = np.array(X).reshape(len(X), -1)
    meta = pd.DataFrame().from_dict(meta)
    
    # embed the features using UMAP
    project = umap.UMAP()
    embed = project.fit_transform(X)
    
    # cluster the embedings (min_cluster_size and min_samples parameters need to be tuned)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=10).fit(embed) # au depart 50 et 3
    meta['cluster'] = clusterer.labels_
    
    # display information for the user to check whether the clustering has gone well or not
    print('clusters for '+c['id'])
    print(meta.groupby('cluster').agg({'fn':'nunique', 'offset':'count'})\
          .rename(columns={'fn':'Number of files in cluster', 'offset':'Number of samples in cluster'}))
    plt.scatter(embed[:,0], embed[:,1], c=clusterer.labels_ , cmap="Paired", s=1)
    plt.show()
    plt.savefig('scatter_'+c['id'])
    plt.close()
    
    # For each cluster, we create a folder of png spectrograms
    for cluster, grp in meta.groupby('cluster'):
        # if the cluster is more than 300 samples big, we consider it is noise (to tune depending on dataset size)
        if len(grp) > 300: 
            continue
        
        # create the cluster folder in the config folder
        os.system('mkdir -p pngs/'+c['id']+'/'+str(cluster))
        
        # for each sample in the cluster, we plot the spectrogram for quick visualisation of the clustering result
        for row in tqdm(grp.itertuples(), desc=str(cluster), total=len(grp), leave=False):
            
            # load the spectrogram from the .npy file
            spectro = np.load(folder+row.fn, allow_pickle=True).item()[c['id']]
            plt.imshow(np.log10(spectro[:, row.offset : row.offset+chunksize]), origin='lower', aspect='auto')
            
            # TODO convert the offset (in max pooled time bins) to seconds
            plt.savefig('pngs/'+c['id']+'/'+str(cluster)+'/'+r.fn[:-9]+'_'+str(r.offset))
            plt.close()
