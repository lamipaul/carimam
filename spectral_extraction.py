import torch
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm
import os
from scipy import signal
from scipy.ndimage.filters import maximum_filter
import soundfile as sf
import pandas as pd
import argparse


"""
This file computes several spectrograms with a given set of paramters.
Spectrograms of size 128x128 will be saved in .npy files for each found sounfiles found in a given folder
"""
winsize = 1024 # global STFT window size (we change the sample rate to tune the freq / time resolutions)

def create_mel_filterbank(sample_rate, frame_len, num_bands, min_freq, max_freq):
    """
    Creates a mel filterbank of `num_bands` triangular filters, with the first
    filter starting at `min_freq` and the last one stopping at `max_freq`.
    Returns the filterbank as a matrix suitable for a dot product against
    magnitude spectra created from samples at a sample rate of `sample_rate`
    with a window length of `frame_len` samples.
    """
    # array of triangular filters' peaks (linear scale)
    peaks_mel = np.linspace(1127*np.log1p(min_freq/700), 1127*np.log1p(max_freq/700), num_bands+2)
    # array of triangular filters' peaks (logarithmic scale)
    peaks_hz = 700 * (np.expm1(peaks_mel / 1127))
    peaks_bin = peaks_hz * frame_len / sample_rate

    # spectrogram freq bins indices
    x = np.arange(frame_len//2 + 1).reshape(-1, 1)

    # build triangular filters from left and right boundaries
    l, c, r = peaks_bin[0:-2], peaks_bin[1:-1], peaks_bin[2:]
    tri_left = (x - l) / (c - l)
    tri_right = (x - r) / (c - r)
    tri = np.minimum(tri_left, tri_right)

    # clip at zero, and normalize each filter by its area
    filterbank = np.maximum(tri, 0)
    filterbank /= filterbank.sum(axis=0)

    # return a weight transformation matrix of shape (num_bands, frame_len//2 + 1)
    return filterbank.T

"""
Set of spectrogram configurations (sampling rate, wether we yield a regular or mel spectrogram, mel start frequency),
each tuned to focus on a given type of vocalisations. e.g. BBF configuration samples at 2kHz.
The output mel spectrogram will thus range from 0 to 1kHz, with 512 logarithmic spaced freq bins.
"""
configs = {
    'BBF': {'fs': 2000, 'mel':True, 'nomel':False, 'melstart':0}, # tuned for fin whale vocs (~60Hz stationnary)
    'HBF': {'fs': 16000, 'mel':True, 'nomel':False, 'melstart':0}, # tuned for humpback whale vocs (~500Hz stationnary)
    'BMF': {'fs': 64000, 'mel':True, 'nomel':False, 'melstart':2000}, # tuned for clicks (sperm whales, delfinids) (~5-30kHz transitory)
    'HMF': {'fs': 256000 , 'mel':True, 'nomel':True, 'melstart':8000}, # tuned for HF clicks (deflinids, ziphius)
    'HF': {'fs': 256000, 'mel':False, 'nomel':True} # tuned for very HF clicks (/!\ for this config, we use a 256pts STFT window)
}
# build a low pass filter used before resampling, and the melbank if needed
for id, c in configs.items():
    if c['mel']:
        c['melbank'] = create_mel_filterbank(c['fs'], winsize, 128, c['melstart'], c['fs']//2)

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if len(batch) > 0 else None

class Dataset(torch.utils.data.Dataset):
    def __init__(self, list, length, fs):
        super(Dataset).__init__()
        self.list, self.length, self.fs = list, length, fs
    def __getitem__(self, idx):
        fn = self.list.iloc[idx]
        if os.path.isfile(outfolder+fn.rsplit('.', 1)[0]+'_spec.npy'):
            return None
        try:
            sig, fs = sf.read(folder+fn)
            if len(sig) < self.length*self.fs:
                return None
            return sig[:self.length*self.fs], fn
        except:
            return None
    def __len__(self):
        return len(self.list)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='blabla')
    parser.add_argument('folder', type=str, )
    args = parser.parse_args()

    folder = '../../DATA/'+args.folder+'/' # path to a given recording station folder
    assert os.path.isdir(folder), "wrong folder name "+args.folder
    outfolder = '../../results/'+args.folder+'/'
    os.system('mkdir -p '+outfolder)

    print('doing ', folder)
    #get filenames list, filter wav files only (possibly sample a subset randomly for testing)
    fns = pd.Series(os.listdir(folder))
    fns = fns[fns.str.endswith('WAV')] #.sample(500)
    source_fs = sf.info(folder+fns.iloc[0]).samplerate
    length = 50
    loader = torch.utils.data.DataLoader(Dataset(fns, length, source_fs), batch_size=8, num_workers=12, collate_fn=collate_fn)
    gpu = torch.device('cuda')
    for id, c in configs.items():
        if c['fs'] < source_fs:
            c['sos'] = signal.butter(3, c['fs']/source_fs, 'lp', output='sos')
        if c['mel']:
            c['melbank'] = torch.Tensor(c['melbank']).to(gpu)
        c['time_uds'] = int(((length*c['fs'] - 1024)/512 +1)//128) if id != 'HF' else int(((length*c['fs'] - 256)/128 +1)//128)
        c['maxpool'] = torch.nn.MaxPool1d((c['time_uds'],))

    with torch.no_grad():
        for batch in tqdm(loader):
            if batch is None:
                continue
            sigs, fns = batch
            # we build a dictionnary containing a spectrogram for each configuration
            out = [{} for i in range(len(sigs))]
            for id, c in configs.items():
                # low pass filter at next nyquist frequency and undersample the signal
                if c['fs'] < source_fs:
                    csig = signal.sosfiltfilt(c['sos'], sigs, axis=-1)
                    csig = torch.Tensor(csig[:,::(source_fs//c['fs'])].copy()).to(gpu)
                else :
                    csig = sigs.float().to(gpu)
                # compute the magnitude spectrogram using the STFT
                if id != 'HF':
                    spec = torch.stft(csig, n_fft=winsize, hop_length=winsize//2, return_complex=False)
                else: # special winsize for HF
                    spec = torch.stft(csig, n_fft=256, hop_length=128, return_complex=False)
                spec = spec.norm(p=2, dim=-1)
                # we undersample the spectrogram over the time dimension to get 128 time bins only
                if c['nomel']:
                    maxpooled = c['maxpool'](spec)[:, :128].cpu().detach()
                    for o, m in zip(out, maxpooled):
                        o['stft_'+id] = m
                if c['mel']:
                    maxpooled = c['maxpool'](torch.matmul(c['melbank'], spec)).cpu().detach()
                    for o, m in zip(out, maxpooled):
                        o['mel_'+id] = m
            # save the dictionnary of spectrograms with at the soundfile location, with the input filename + '_spec.npy'
            for o, fn in zip(out, fns):
                np.save(outfolder+fn.rsplit('.', 1)[0]+'_spec.npy', o)
