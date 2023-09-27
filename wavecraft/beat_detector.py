import librosa
import numpy as np
import argparse
import scipy
import os
from wavecraft.utils import colors

class BeatDetector:
    def __init__(self, args):
        self.input_file = args.input_file
        self.sr = args.sample_rate
        self.hop_size = args.hop_size
        self.n_fft = args.n_fft
        self.n_chroma = 12
        self.n_bands =  6
        self.fmin = args.fmin
        self.fmax = args.fmax
        self.k = args.k
        # self.args.output_directory = self.args.output_directory or os.path.splitext(self.args.input_file)[0] + '_segments'
        

    def main(self):
        y, sr = librosa.load(self.input_file, sr=self.sr)
        
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_size, trim=False)
        
        y_padded = np.pad(y, (self.n_fft // 2, self.n_fft // 2), mode='reflect')
        print(librosa.frames_to_time(beat_frames, sr=sr))
        
        chroma = librosa.feature.chroma_cqt(y=y_padded, sr=sr, n_chroma=self.n_chroma, hop_length=self.hop_size, fmin=self.fmin)
        chroma_sync = librosa.util.sync(chroma, beat_frames, aggregate=np.median)

        energy = librosa.feature.rms(y=y_padded, frame_length=self.n_fft, hop_length=self.hop_size)
        energy_sync = librosa.util.sync(energy, beat_frames, aggregate=np.median)

        contrast = librosa.feature.spectral_contrast(y=y_padded, sr=sr,
                                                     n_fft=self.n_fft, hop_length=self.hop_size,
                                                     n_bands=self.n_bands)
        contrast_sync = librosa.util.sync(contrast, beat_frames, aggregate=np.median)

        centroid = librosa.feature.spectral_centroid(y=y_padded, sr=sr)
        centroid_sync = librosa.util.sync(centroid, beat_frames, aggregate=np.median)

        zcr = librosa.feature.zero_crossing_rate(y=y_padded, frame_length=self.n_fft, hop_length=self.hop_size)
        zcr_sync = librosa.util.sync(zcr, beat_frames, aggregate=np.median)

        flatness = librosa.feature.spectral_flatness(y=y_padded, n_fft=self.n_fft, hop_length=self.hop_size)
        flatness_sync = librosa.util.sync(flatness, beat_frames, aggregate=np.median)

        # Combine all features into a single matrix
        features = np.vstack([chroma_sync, energy_sync, contrast_sync, centroid_sync, zcr_sync, flatness_sync])
        features_norm = librosa.util.normalize(features, norm=2, axis=0)

        # Compute the distance matrix
        dist = librosa.segment.recurrence_matrix(features_norm, mode='affinity', sym=True)
        labels = librosa.segment.agglomerative(dist, self.k)
        
        boundaries = 1 + np.flatnonzero(labels[:-1] != labels[1:])
        boundaries = librosa.util.fix_frames(boundaries, x_min=0)
        bound_frames = beat_frames[boundaries]
        bound_frames = librosa.util.fix_frames(bound_frames, x_min=0, x_max=chroma.shape[1]-1)
        bound_times = librosa.frames_to_time(bound_frames, sr=sr, hop_length=self.hop_size)
        
        print(f'{colors.GREEN}Detected {len(bound_frames)} beats.{colors.ENDC}')
        print(f'{colors.GREEN}Beat times: {bound_times}{colors.ENDC}')
        
        return bound_frames
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect beats in an audio file.")
    parser.add_argument("--i", "--input-file", type=str, help="Path to the audio file (wav, aif, aiff).", required=True)
    parser.add_argument("--n-fft", type=int, default=2048, help="FFT size. Default is 2048.", required=False)
    parser.add_argument("--hop-size", type=int, default=512, help="Hop size. Default is 512.", required=False)
    parser.add_argument("--n-bands", type=int, default=6, help="Number of spectral contrast bands. Default is 6.", required=False)
    parser.add_argument("--fmin", type=float, default=0, help="Minimum frequency. Default is 0.", required=False)
    parser.add_argument("--fmax", type=float, default=16000, help="Maximum frequency. Default is 16000.", required=False)
    parser.add_argument("--sample-rate", type=int, default=48000, help="Sample rate of the audio file. Default is 48000.", required=False)
    parser.add_argument("--k", type=int, default=5, help="Number of beat clusters. Default is 5.", required=False)
    args = parser.parse_args()

    detector = BeatDetector(args)
    boundaries = detector.main()

    print(boundaries) 