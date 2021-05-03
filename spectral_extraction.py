import numpy as np
import torch
from tqdm import tqdm
import os
from scipy import signal
from scipy.ndimage.filters import maximum_filter
import soundfile as sf
import pandas as pd

folder = '../DATA/BONAIRE/session1_20201217to20210126/'
winsize=1024
source_fs = 512000

def create_mel_filterbank(sample_rate, frame_len, num_bands, min_freq, max_freq):
    peaks_mel = np.linspace(1127*np.log1p(min_freq/700), 1127*np.log1p(max_freq/700), num_bands+2)
    peaks_hz = 700 * (np.expm1(peaks_mel / 1127))
    peaks_bin = peaks_hz * frame_len / sample_rate
    x = np.arange(frame_len//2 + 1).reshape(-1, 1)
    l, c, r = peaks_bin[0:-2], peaks_bin[1:-1], peaks_bin[2:]
    tri_left = (x - l) / (c - l)
    tri_right = (x - r) / (c - r)
    tri = np.minimum(tri_left, tri_right)
    filterbank = np.maximum(tri, 0)
    filterbank /= filterbank.sum(axis=0)
    return filterbank.T


configs = [
{'id':'BBF', 'fs': 2000, 'mel':True, 'nomel':False, 'melstart':0},
{'id':'HBF', 'fs': 16000, 'mel':True, 'nomel':False, 'melstart':0},
{'id':'BMF', 'fs': 64000, 'mel':True, 'nomel':False, 'melstart':2000},
{'id':'HMF', 'fs': 256000 , 'mel':True, 'nomel':True, 'melstart':8000},
{'id':'HF', 'fs': 256000, 'mel':False, 'nomel':True}
]
for c in configs:
    if c['mel']:
        c['melbank'] = create_mel_filterbank(c['fs'], winsize, 128, c['melstart'], c['fs']//2)
    c['sos'] = signal.butter(3, c['fs']/source_fs, 'lp', output='sos')


print('doing ', folder)
#get filenames list
fns = pd.Series(os.listdir(folder))
fns = fns[fns.str.endswith('WAV')] #.sample(500)

for fn in tqdm(fns):
    try:
        sig, fs = sf.read(folder+fn)
    except:
        print('failed with ',fn)
        continue
    out = {}
    for c in configs:
        csig = signal.sosfiltfilt(c['sos'], sig)
        csig = csig[::(fs//c['fs'])]
        if c['id'] != 'HF':
            f, t, spec = signal.stft(csig, fs=c['fs'], nperseg=winsize, noverlap=winsize//2)
        else: # special winsize for HF
            f, t, spec = signal.stft(csig, fs=c['fs'], nperseg=256, noverlap=128)
        spec = np.abs(spec)
        time_uds = spec.shape[1]//128
        if c['nomel']:
            out['stft_'+c['id']] = maximum_filter(spec, (1, time_uds))[:128, ::time_uds]
        if c['mel']:
            out['mel_'+c['id']] = maximum_filter(np.matmul(c['melbank'], spec), (1, time_uds))[:,::time_uds]
    np.save(folder+fn.rsplit('.', 1)[0]+'_spec.npy', out)
