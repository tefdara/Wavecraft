import librosa
import numpy as np

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
        

    def main(self):
        y, sr = librosa.load(self.input_file, sr=self.sr)
        
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_size, trim=False)
        
        y_padded = np.pad(y, (self.n_fft // 2, self.n_fft // 2), mode='reflect')
        
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
        
        return bound_frames
