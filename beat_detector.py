import librosa
import numpy as np
import argparse

class BeatDetector:
    def __init__(self, args):
        self.args = args

    def detect_beats(self):
        y, sr = librosa.load(self.args.input_file, sr=self.args.sample_rate)

        # Compute the tempo and beat frames
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=int(self.args.hop_length_seconds * sr),
                                                      start_bpm=self.args.tempo, n_fft=self.args.fft_size,
                                                      hop_length=self.args.hop_size, n_mels=self.args.n_mels,
                                                      fmin=self.args.fmin, fmax=self.args.fmax)

        # Convert beat frames to sample indices
        beat_samples = librosa.frames_to_samples(beat_frames, hop_length=int(self.args.hop_length_seconds * sr))

        # Add padding to the beginning and end of the audio signal
        y_padded = np.pad(y, (self.args.n_fft // 2, self.args.n_fft // 2), mode='reflect')

        # Compute the beat-synchronous chroma features
        chroma = librosa.feature.chroma_cqt(y=y_padded, sr=sr, hop_length=int(self.args.hop_length_seconds * sr),
                                            n_fft=self.args.n_fft, hop_length=self.args.hop_size,
                                            n_chroma=self.args.n_chroma, fmin=self.args.fmin, fmax=self.args.fmax)
        chroma_sync = librosa.util.sync(chroma, beat_frames, aggregate=np.median)

        # Compute the beat-synchronous onset strength envelope
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=int(self.args.hop_length_seconds * sr),
                                                 n_fft=self.args.fft_size, hop_length=self.args.hop_size,
                                                 fmin=self.args.fmin, fmax=self.args.fmax)
        onset_env_sync = librosa.util.sync(onset_env, beat_frames, aggregate=np.median)

        # Compute the beat-synchronous energy envelope
        energy = librosa.feature.rms(y=y_padded, frame_length=self.args.n_fft, hop_length=self.args.hop_size)
        energy_sync = librosa.util.sync(energy, beat_frames, aggregate=np.median)

        # Compute the beat-synchronous spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y_padded, sr=sr, hop_length=int(self.args.hop_length_seconds * sr),
                                                     n_fft=self.args.n_fft, hop_length=self.args.hop_size,
                                                     n_bands=self.args.n_bands, fmin=self.args.fmin, fmax=self.args.fmax)
        contrast_sync = librosa.util.sync(contrast, beat_frames, aggregate=np.median)

        # Compute the beat-synchronous spectral centroid
        centroid = librosa.feature.spectral_centroid(y=y_padded, sr=sr, hop_length=int(self.args.hop_length_seconds * sr),
                                                     n_fft=self.args.n_fft, hop_length=self.args.hop_size,
                                                     fmin=self.args.fmin, fmax=self.args.fmax)
        centroid_sync = librosa.util.sync(centroid, beat_frames, aggregate=np.median)

        # Compute the beat-synchronous spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=y_padded, sr=sr, hop_length=int(self.args.hop_length_seconds * sr),
                                                       n_fft=self.args.n_fft, hop_length=self.args.hop_size,
                                                       fmin=self.args.fmin, fmax=self.args.fmax)
        bandwidth_sync = librosa.util.sync(bandwidth, beat_frames, aggregate=np.median)

        # Compute the beat-synchronous zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=y_padded, frame_length=self.args.n_fft, hop_length=self.args.hop_size)
        zcr_sync = librosa.util.sync(zcr, beat_frames, aggregate=np.median)

        # Compute the beat-synchronous spectral flatness
        flatness = librosa.feature.spectral_flatness(y=y_padded, n_fft=self.args.n_fft, hop_length=self.args.hop_size)
        flatness_sync = librosa.util.sync(flatness, beat_frames, aggregate=np.median)

        # Combine all features into a single matrix
        features = np.vstack([chroma_sync, onset_env_sync, energy_sync, contrast_sync, centroid_sync, bandwidth_sync, zcr_sync, flatness_sync]).T

        # Normalize the features
        features_norm = librosa.util.normalize(features, norm=2, axis=0)

        # Compute the distance matrix
        dist = librosa.segment.recurrence_matrix(features_norm, mode='affinity', metric='cosine', sym=True)

        # Compute the beat-synchronous segmentation
        _, labels = librosa.segment.agglomerative(dist, n_segments=None, linkage='ward', return_distance=False)

        # Convert the labels to segment boundaries
        boundaries = librosa.util.fix_frames(librosa.segment.label2seg(labels, frame_length=self.args.hop_length_seconds * sr, hop_length=self.args.hop_length_seconds * sr))

        return boundaries
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect beats in an audio file.")
    parser.add_argument("--i", "--input-file", type=str, help="Path to the audio file (wav, aif, aiff).", required=True)
    parser.add_argument("--tempo", type=float, default=None, help="Tempo of the audio file. Optional.", required=False)
    parser.add_argument("--hop-length-seconds", type=float, default=0.01, help="Hop length in seconds. Default is 0.01s.", required=False)
    parser.add_argument("--fft-size", type=int, default=2048, help="FFT size. Default is 2048.", required=False)
    parser.add_argument("--n-fft", type=int, default=2048, help="FFT size. Default is 2048.", required=False)
    parser.add_argument("--hop-size", type=int, default=512, help="Hop size. Default is 512.", required=False)
    parser.add_argument("--n-mels", type=int, default=128, help="Number of Mel bands. Default is 128.", required=False)
    parser.add_argument("--n-chroma", type=int, default=12, help="Number of chroma bins. Default is 12.", required=False)
    parser.add_argument("--n-bands", type=int, default=6, help="Number of spectral contrast bands. Default is 6.", required=False)
    parser.add_argument("--fmin", type=float, default=0, help="Minimum frequency. Default is 0.", required=False)
    parser.add_argument("--fmax", type=float, default=None, help="Maximum frequency. Optional.", required=False)
    parser.add_argument("--sample-rate", type=int, default=44100, help="Sample rate of the audio file. Default is 44100.", required=False)

    args = parser.parse_args()

    detector = BeatDetector(args)
    boundaries = detector.detect_beats()

    print(boundaries) 