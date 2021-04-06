import glob
import os
import argparse

import pandas as pd
import librosa
from tqdm import tqdm
import soundfile as sf
import json


def clean_digits(digit):
    """
    gets the numbers out of a string
    """
    digit = [c for c in digit if c.isdigit()]
    return ''.join(digit)

def load_intervals(path):
    """
    Extracts time intervals from a transcription.
    """
    intervals = []
    num_syl = []
    with open (path, "r") as myfile:
        lines = myfile.readlines()
    for index, line in enumerate(lines):
        if line.startswith('*CHI:'):
            #extract a single time interval from a line
            interval = line.split('\t')[-1].strip()
            interval = interval.split('_')
            interval = [int(clean_digits(i.strip())) for i in interval]
            intervals.append(interval)
            num_syl.append(clean_digits(list(lines)[index + 1].strip()))
    return intervals, num_syl

def main():

    parser = argparse.ArgumentParser(description='Extract cutouts from wav-files')

    parser.add_argument('--indir', type=str, default='data/orig/',
                        help='location of the original children folders')
    parser.add_argument('--outdir', type=str, default='data/wav/',
                        help='location of the output folder for the extractions')

    args = parser.parse_args()

    with open('hearing.json') as f:
        status = json.loads(f.read())

    with open('camera.json') as f:
        camera_ = json.loads(f.read())
    camera = {}
    for child in camera_:
        camera[child[:3].lower()] = camera_[child]

    audio_files = glob.glob(args.indir + '**/*.wav', recursive=True)
    cha_files = [f.replace('.wav', '.cha').lower() for f in audio_files]

    metadata, utterance_idx = [], 0
    for audio_f, cha_f in tqdm(list(zip(audio_files, cha_files))):
        signal, sr = librosa.load(path=audio_f, sr=None, mono=True)
        intervals, num_syl = load_intervals(cha_f)
        child = audio_f.split(os.sep)[-2].lower()[:3]
        fn = os.path.basename(audio_f).replace('.wav', '')
        age = str(int(fn[3:5]) * 12 + int(fn[5:7]))
        for interval, ns in zip(intervals, num_syl):
            length = float(interval[1] - interval[0])
            if (length < 0) or (length > 30000):
                continue
            segment = signal[int(interval[0] * sr / 1000): int(interval[1] * sr / 1000)]
            cutout_fn = str(utterance_idx).rjust(7, '0') + '.wav'
            sf.write(file=args.outdir + cutout_fn, data=segment, samplerate=sr)
            metadata.append((cutout_fn, child, status[child], \
                             age, camera[child], length, ns, \
                             float(ns) / (float(length) / 1000)))
            utterance_idx += 1

    cols = ('filename', 'child', 'status', 'age', 'camera', 'length', 'num syl', 'speak rate')
    df = pd.DataFrame(metadata, columns=cols)
    df.to_csv(args.outdir + '../metadata.csv', header=True, index=False)


if __name__ == '__main__':
    main()
