from tqdm import tqdm
import numpy as np
from scipy import signal
import os
import matplotlib.pyplot as plt
import soundfile as sf
from spectral_extraction import configs

sons = '../../DATA/GUA_BREACH/session2_20210208_20210322/'
configID = 'mel_HBF'
pngs = '../pngs_GUA_BREACH/'+configID+'/'
goodClusters = np.array([17, 15, 6, 4, 2]).astype(str)
chunksize = 20 # same as in sort_cluster.py, not in seconds
winsize = 1024 # global STFT window size (we change the sample rate to tune the freq / time resolutions)

config = configs[configID.split('_')[1]]
mfs = config['fs']
fscale =  700* np.expm1(np.linspace(1127*np.log1p(config['melstart']/700), 1127*np.log1p(mfs/2/700), 128+2) / 1127)[1:-1]

for cluster in goodClusters:
    for fn in tqdm(os.listdir(pngs+cluster), desc='doing cluster '+cluster):
        if fn.startswith('spectro'):
            continue
        offset = int(fn.split('_')[-1][:-4])
        wavfn = fn.rsplit('_', 1)[0] + '.WAV'
        fileDur = sf.info(sons+wavfn).duration
        source_fs = sf.info(sons+wavfn).samplerate
        timeuds = (fileDur * mfs // 128 +1)//128 # == len(spectro) // 128
        start = offset * timeuds * 128 / mfs # in seconds, idtimebin * hopsize / fs
        stop = (offset + chunksize) * timeuds * 128 / mfs
        sig, fs = sf.read(sons+wavfn, start=int(start * source_fs), stop=int(stop * source_fs))
        sig = signal.resample(sig, int((stop-start)*mfs))
        f, t, spec = signal.stft(sig, fs=mfs, nperseg=winsize, noverlap=winsize//2)
        spec = np.abs(spec)
        if 'mel' in config:
            spec = np.matmul(config['melbank'], spec)
        plt.imshow(10*np.log10(spec),  extent=[0, t[-1], 0, f[-1]], origin='lower', aspect='auto')
        plt.savefig(pngs+cluster+'/spectro_'+fn.rsplit('_',1)[0]+'_{:.0f}'.format(start))
        plt.close()
