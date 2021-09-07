import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy import signal
import os
import matplotlib.pyplot as plt
import soundfile as sf
from spectral_extraction import configs
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='blabla')
parser.add_argument('folder', type=str)
parser.add_argument('pickle_filename', type=str)
parser.add_argument('--max_spectros', type=int, default=50)
parser.add_argument('--goodClusters', nargs='+', required=True, type=int)
args = parser.parse_args()
df_filename = args.pickle_filename
goodClusters = list(args.goodClusters)
chunksize= int(df_filename.split('/')[-1].split('_')[3][9:])
configID = df_filename.split('/')[-1].split('_')
configID = configID[1]+'_'+configID[2]
sons = '../../DATA/'+args.folder+'/'
outfolder = '../pngs/'+args.folder+'/'
assert configID in ['mel_BBF', 'mel_HBF', 'mel_BMF', 'mel_HMF', 'stft_HMF', 'stft_HF'], "wrong config name "+configID

chunksize = 5 # same as in sort_cluster.py, not in seconds
winsize = 1024 if configID != 'stft_HF' else 256 # global STFT window size (we change the sample rate to tune the freq / time resolutions)
hopsize = winsize//2

config = configs[configID.split('_')[1]]
mfs = config['fs']
#fscale =  700* np.expm1(np.linspace(1127*np.log1p(config['melstart']/700), 1127*np.log1p(mfs/2/700), 128+2) / 1127)[1:-1]

df = pd.read_pickle(outfolder+df_filename)

for cluster, grp in df[df.cluster.isin(goodClusters)].groupby('cluster'):
    grp = grp.sample(args.max_spectros) if args.max_spectros < len(grp) else grp
    os.system('mkdir -p '+outfolder+str(cluster))
    #for fn in tqdm(os.listdir(pngs+cluster), desc='doing cluster '+cluster):
    for i, r in tqdm(grp.iterrows(), desc='doing cluster '+str(cluster), total=len(grp)):
        info = sf.info(sons+r.wavfn)
        fileDur, source_fs = info.duration, info.samplerate
        timeuds = ((fileDur * mfs - winsize) // hopsize +1)//128 # == len(spectro) // 128
        start = r.offset * timeuds * hopsize / mfs # in seconds, idtimebin * hopsize / fs
        stop = (r.offset + chunksize) * timeuds * hopsize / mfs
        sig, fs = sf.read(sons+r.wavfn, start=int(start * source_fs), stop=int(stop * source_fs))
        if mfs != source_fs:
            sig = signal.resample(sig, int((stop-start)*mfs))
        f, t, spec = signal.stft(sig, fs=mfs, nperseg=winsize, noverlap=winsize - winsize//4)
        spec = np.abs(spec)
        if 'mel' in configID:
            spec = np.matmul(config['melbank'], spec)
        plt.imshow(10*np.log10(spec),  extent=[0, t[-1], 0, f[-1]], origin='lower', aspect='auto')
        plt.savefig(outfolder+str(cluster)+'/spectro_'+r.wavfn.split('.')[0]+'_{:.0f}'.format(start))
        plt.close()
