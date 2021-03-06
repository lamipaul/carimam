from scipy import signal
from tqdm import tqdm
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import pandas as pd
import os
import umap
import hdbscan
import argparse
from spectral_extraction import configs
try:
    import sounddevice as sd
    soundAvailable = True
except:
    soundAvailable = False

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='blabla')
parser.add_argument('folder', type=str)
parser.add_argument('config', type=str)
parser.add_argument('--sample_size', type=int, default=None)
parser.add_argument('--chunk_size', type=int, default=5) # size (in time bins) of chunks of spectrogram to cluster independantly
parser.add_argument('--min_cluster_size', type=int, default=50)
parser.add_argument('--min_sample', type=int, default=10)
parser.add_argument('--quantiles', nargs='+', type=int, default=[])
args = parser.parse_args()
chunksize= args.chunk_size
config = args.config
quantiles = list(args.quantiles)
specfolder = '../../results/'+args.folder+'/' # path to a given recording station folder
assert os.path.isdir(specfolder), "wrong folder name "+args.folder
wavfolder = '../../DATA/'+args.folder+'/'
outfolder = '../pngs/'+args.folder+'/'
os.system('mkdir -p '+outfolder)
assert config in ['mel_BBF', 'mel_HBF', 'mel_BMF', 'mel_HMF', 'stft_HMF', 'stft_HF'], "wrong config name"

sort = True # whether we sort each frequency bins by descending order (used to build an energy distribution image and eliminate time dependent features)

# get filenames list, filter _spec.npy files only (possibly sample a subset randomly for testing)
fns = pd.Series(os.listdir(specfolder)).reset_index(drop=True)
fns = fns[fns.str.endswith('_spec.npy')] if args.sample_size is None else fns[fns.str.endswith('_spec.npy')].sample(args.sample_size)

df_filename = outfolder+'clusters_'+config+'_chunksize'+str(chunksize)+('_sorted_' if sort else '')+('_quant'+''.join(map(str, quantiles)) if len(quantiles)>0 else '')+'.pkl'

if not os.path.isfile(df_filename) :
    # arrays X and meta will hold features and metadata for each samples to be projected / clustered
    X, meta = [], []
    for f in tqdm(fns, desc='loading spectros for '+config, leave=False):
        # load the spectrogram from the .npy file
        dic =  np.load(specfolder+f, allow_pickle=True,protocol=4).item()
        if not config in dic:
            continue
        spectro = dic[config]

        # cut the spectrogram in chunks time wise
        for offset in np.arange(0, spectro.shape[1]-chunksize, chunksize):
            # get logarithmic magnitude
            temp = np.log10(spectro[:,offset:offset+chunksize])
            if sort :
                # if sort is True, we sort each frequency bin in descending order
                temp = np.flip(np.sort(temp, axis=1), axis=1)
                # we can then select a subset of bins (similar to quantiles)
                if len(quantiles) == 0:
                    X.append(temp)
                else : 
                    X.append(temp[:, quantiles])
            else :
                # else, we use the whole spectrogram as input features for projection
                X.append(temp)
            # save filename and offset to retrieve the sample later on
            meta.append({'fn':f, 'offset':offset})
    print('done loading spectros, doing projection...')
    # rearange X and meta arrays for easier later use
    X = np.array(X).reshape(len(X), -1)
    meta = pd.DataFrame().from_dict(meta)
    # embed the features using UMAP
    project = umap.UMAP()
    embed = project.fit_transform(X)
    meta['umap_x'] = embed[:,0]
    meta['umap_y'] = embed[:,1]
    meta['wavfn'] = meta.fn.str[:-9]+'.WAV'
    meta.to_pickle(df_filename)
else :
    print('loading previous projection')
    meta = pd.read_pickle(df_filename)
    embed = meta[['umap_x', 'umap_y']].to_numpy()

# cluster the embedings (min_cluster_size and min_samples parameters need to be tuned)
clusterer = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size, min_samples=args.min_sample).fit(embed)
meta['cluster'] = clusterer.labels_
meta.to_pickle(df_filename)
configDic = configs[config.split('_')[1]]
mfs = configDic['fs']
winsize = 1024 if config != 'stft_HF' else 256 # global STFT window size (we change the sample rate to tune the freq / time resolutions)
hopsize = winsize//2

# display information for the user to check whether the clustering has gone well or not
print('clusters for '+config)
print(meta.groupby('cluster').agg({'fn':'nunique', 'offset':'count'})\
        .rename(columns={'fn':'n unique files', 'offset':'n samples'}))
figscat = plt.figure()
plt.scatter(embed[:,0], embed[:,1], c=clusterer.labels_ , cmap="tab20" if len(np.unique(clusterer.labels_))>10 else "tab10", s=1)
plt.title(args.folder)
plt.colorbar()
axScat = figscat.axes[0]
figSpec = plt.figure()
plt.scatter(0, 0)
axSpec = figSpec.axes[0]

def onclick(event):
    #get row
    left, right, bottom, top = axScat.get_xlim()[0], axScat.get_xlim()[1], axScat.get_ylim()[0], axScat.get_ylim()[1]
    rangex, rangey =  right - left, top - bottom
    diff = abs(meta.umap_x - event.xdata)/rangex + abs(meta.umap_y  - event.ydata)/rangey
    row = meta.loc[diff.idxmin()]
    info = sf.info(wavfolder+row.wavfn)
    fs, fileDur = info.samplerate, info.duration 
    timeuds = ((fileDur * mfs - winsize) // hopsize +1)//128 # == len(spectro) // 128
    start = row.offset * timeuds * hopsize / mfs # in seconds, idtimebin * hopsize / fs
    stop = (row.offset + chunksize) * timeuds * hopsize / mfs
    sig, fs = sf.read(wavfolder+row.wavfn, start=int(start*fs), stop=int(stop*fs))
    if mfs != fs:
        sig = signal.resample(sig, int((stop-start)*mfs))
    f, t, spec = signal.stft(sig, fs=mfs, nperseg=winsize, noverlap=winsize-winsize//4)
    spec = np.abs(spec)
    if 'mel' in config:
        spec = np.matmul(configDic['melbank'], spec)
        f =  700* np.expm1(np.linspace(1127*np.log1p(configDic['melstart']/700), 1127*np.log1p(mfs/2/700), 128+2) / 1127)[1:-1]
    axSpec.imshow(10*np.log10(spec),  extent=[0, t[-1], f[0], f[-1]], origin='lower', aspect='auto')
    axSpec.set_title(row.wavfn+' cluster '+str(row.cluster)+'\nstart (timebins) : '+str(row.offset)+' start (sec):'+str(round(start, 1)))
    axScat.scatter(row.umap_x, row.umap_y, c='r')
    axScat.set_xlim(left, right)
    axScat.set_ylim(bottom, top)
    figSpec.canvas.draw()
    figscat.canvas.draw()
    if soundAvailable:
        sd.play(sig*10, mfs)

cid = figscat.canvas.mpl_connect('button_press_event', onclick)

plt.show()
figscat.savefig('scatter_'+config)
plt.close()
