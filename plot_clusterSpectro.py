import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy import signal
import os
import matplotlib.pyplot as plt
import soundfile as sf
from spectral_extraction import configs

sons = '../../DATA/LOT2/JAM_20210406_20320510/'
configID = 'stft_HF'
pngs = '../pngs_JAM_20210406_20320510/'+configID+'/'
goodClusters = [0, 1]
chunksize = 20 # same as in sort_cluster.py, not in seconds
winsize = 1024 if configID != 'stft_HF' else 256 # global STFT window size (we change the sample rate to tune the freq / time resolutions)
hopsize = winsize//2

config = configs[configID.split('_')[1]]
mfs = config['fs']
#fscale =  700* np.expm1(np.linspace(1127*np.log1p(config['melstart']/700), 1127*np.log1p(mfs/2/700), 128+2) / 1127)[1:-1]

df = pd.read_pickle(pngs+'clusters_'+configID+'.pkl')

for cluster, grp in df[df.cluster.isin(goodClusters)].groupby('cluster'):
    #for fn in tqdm(os.listdir(pngs+cluster), desc='doing cluster '+cluster):
    for i, r in tqdm(grp.iterrows(), desc='doing cluster '+str(cluster), total=len(grp)):
        wavfn = r.fn.rsplit('_', 1)[0] + '.WAV'
        fileDur = sf.info(sons+wavfn).duration
        source_fs = sf.info(sons+wavfn).samplerate
        timeuds = ((fileDur * mfs - winsize) // hopsize +1)//128 # == len(spectro) // 128
        start = r.offset * timeuds * hopsize / mfs # in seconds, idtimebin * hopsize / fs
        stop = (r.offset + chunksize) * timeuds * hopsize / mfs
        sig, fs = sf.read(sons+wavfn, start=int(start * source_fs), stop=int(stop * source_fs))
        sig = signal.resample(sig, int((stop-start)*mfs))
        f, t, spec = signal.stft(sig, fs=mfs, nperseg=winsize, noverlap=winsize//2)
        spec = np.abs(spec)
        if config['mel']:
            spec = np.matmul(config['melbank'], spec)
        plt.imshow(10*np.log10(spec),  extent=[0, t[-1], 0, f[-1]], origin='lower', aspect='auto')
        plt.savefig(pngs+str(cluster)+'/spectro_'+r.fn.rsplit('_',1)[0]+'_{:.0f}'.format(start))
        plt.close()
