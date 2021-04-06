import argparse
from scipy.signal import spectral
import numpy as np
import librosa
import tqdm
import glob
import os
import shutil


def spectrogram(audio, fft_freq=1024, win_length=1024, 
                hop_length=512, melW=None):
    """
    Creates a spectrogram of the input wave
    """
    ham_win = np.hamming(win_length)
    X = spectral.spectrogram(
        x=audio,
        nfft=fft_freq,
        window=ham_win,
        nperseg=win_length,
        noverlap=win_length - hop_length,
        detrend=False,
        return_onesided=True,
        mode='magnitude')[-1].T
    if melW is not None:
        X = np.dot(X, melW)
    return np.log(X + 1e-8).astype(np.float32)

def extract_spectrograms(directory, fft_freq=1024, win_length=1024,
                         minf=0, maxf=8000, hop_length=512,
                         mel_freq=64, sr=44100):
    """
    Loads all the spectrograms of all utterances.
    """

    melW = librosa.filters.mel(sr=sr, n_fft=fft_freq,
                               n_mels=mel_freq, fmin=minf, 
                               fmax=maxf).T

    for wav_f in tqdm.tqdm(glob.glob(directory + 'wav/*.wav')):
        spec_name = os.path.basename(wav_f)[:-4] + '.npy'
        # if os.path.exists(directory + 'spec/' + spec_name):
        #     print(spec_name, ' already exists')
        #     continue
        y, sr = librosa.load(path=wav_f, sr=None, mono=True)
        try:
            spec = spectrogram(audio=y, 
                               fft_freq=fft_freq, 
                               win_length=win_length, 
                               hop_length=hop_length, 
                               melW=melW)
        except ValueError:
            print(spec_name)
            continue
        np.save(file=directory+'spec/'+spec_name, arr=spec)

def main():
    parser = argparse.ArgumentParser(description='parameters for spectrogram calculation')
    parser.add_argument('--dir', type=str,
                        default='data/',
                        help='Location of the data folder')
    parser.add_argument('--mel_freq', type=int,
                        default=64,
                        help='Number of frequencies in the MEL-spectrogram. Use zero for no MEL')
    parser.add_argument('--fft_freq', type=int,
                        default=1024,
                        help='Number of frequencies used for the STFT')
    parser.add_argument('--hop_length', type=int,
                        default=1024-360,
                        help='Number of elements that one hops by at every iteration')
    parser.add_argument('--win_length', type=int,
                        default=1024,
                        help='number of elements in one STFT frame')
    parser.add_argument('--visualization', type=bool,
                        default=False,
                        help='decides whether to save figures with spectrograms')


    args = parser.parse_args()
    print(args)

    extract_spectrograms(directory=args.dir, 
                         mel_freq=args.mel_freq,
                         win_length=args.win_length, 
                         fft_freq=args.fft_freq,
                         hop_length=args.hop_length)

    all_specs = [os.path.basename(i)[:-4] + '.wav' for i in glob.glob(args.dir + 'spec/')]
    meta_cut = meta.loc[meta['filename'].isin(all_specs)]

if __name__ == '__main__':
    main()
