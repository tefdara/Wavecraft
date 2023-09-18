import argparse
import librosa
import os
import sys
import soundfile as sf
import numpy as np
import traceback
import asyncio
from utils import bcolors, Utils


utils = Utils()
class OnsetDetector:
    def __init__(self, args):
        self.args = args
        self.args.output_directory = self.args.output_directory or os.path.splitext(self.args.input_file)[0] + '_segments'
    
    def compute_onsets(self, y, sr, hop_length=512, n_fft=2048, fmin=27.5, fmax=16000., lag=2, max_size=3, env_method='mel'):
        if env_method == 'mel': 
            print(bcolors.YELLOW + 'Using mel spectrogram for onset detection.' + bcolors.ENDC)
            mel_spectogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=138, fmin=fmin, fmax=fmax)
            S = librosa.power_to_db(mel_spectogram, ref=np.max)
            o_env = librosa.onset.onset_strength(S=S,
                                                  sr=sr,
                                                  hop_length=hop_length,
                                                  lag=lag, max_size=max_size)
        elif env_method == 'rms':
            print(bcolors.YELLOW + 'Using RMS for onset detection.' + bcolors.ENDC)
            S = np.abs(librosa.stft(y=y))
            rms = librosa.feature.rms(S=S, frame_length=n_fft, hop_length=hop_length)
            o_env = librosa.onset.onset_strength(S=rms[0],
                                                  sr=sr,
                                                  hop_length=hop_length,
                                                  lag=lag, max_size=max_size)
        elif env_method == 'cqt':
            print(bcolors.YELLOW + 'Using CQT for onset detection.' + bcolors.ENDC)
            cqt = librosa.cqt(y=y, sr=sr, hop_length=hop_length, fmin=fmin, n_bins=252, bins_per_octave=36)
            S = librosa.amplitude_to_db(cqt, ref=np.max)
            o_env = librosa.onset.onset_strength(S=S,
                                                  sr=sr,
                                                  hop_length=hop_length,
                                                  lag=lag, max_size=max_size)
        elif env_method == 'stft':
            print(bcolors.YELLOW + 'Using STFT for onset detection.' + bcolors.ENDC)
            stft = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)
            S = librosa.amplitude_to_db(stft, ref=np.max)
            o_env = librosa.onset.onset_strength(S=S,
                                                  sr=sr,
                                                  hop_length=hop_length,
                                                  lag=lag, max_size=max_size)
        elif env_method == 'cens':
            print(bcolors.YELLOW + 'Using chroma cens for onset detection.' + bcolors.ENDC)
            cens = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length, n_chroma=12, bins_per_octave=36)
            o_env = librosa.onset.onset_strength(S=cens,
                                                  sr=sr,
                                                  hop_length=hop_length,
                                                  lag=lag, max_size=max_size)
        elif env_method == 'cqt_chr':
            print(bcolors.YELLOW + 'Using CQT chroma for onset detection.' + bcolors.ENDC)
            cqt_chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length, n_chroma=12, bins_per_octave=36)
            o_env = librosa.onset.onset_strength(S=cqt_chroma,
                                                  sr=sr,
                                                  hop_length=hop_length,
                                                  lag=lag, max_size=max_size)
        elif env_method == 'mfcc':
            print(bcolors.YELLOW + 'Using MFCC for onset detection.' + bcolors.ENDC)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13, n_mels=138, fmin=fmin, fmax=fmax)
            o_env = librosa.onset.onset_strength(S=mfcc,
                                                  sr=sr,
                                                  hop_length=hop_length,
                                                  lag=lag, max_size=max_size)
        elif env_method == 'tmpg':
            print(bcolors.YELLOW + 'Using tempogram for onset detection.' + bcolors.ENDC)
            tempogram = librosa.feature.tempogram(y=y, sr=sr, hop_length=hop_length, win_length=384, center=True)
            o_env = librosa.onset.onset_strength(S=tempogram,
                                                  sr=sr,
                                                  hop_length=hop_length,
                                                  lag=lag, max_size=max_size)
        elif env_method == 'ftmpg':
            print(bcolors.YELLOW + 'Using fourier tempogram for onset detection.' + bcolors.ENDC)
            fourier_tempogram = librosa.feature.fourier_tempogram(y=y, sr=sr, hop_length=hop_length, win_length=384, center=True)
            o_env = librosa.onset.onset_strength(S=fourier_tempogram,
                                                  sr=sr,
                                                  hop_length=hop_length,
                                                  lag=lag, max_size=max_size)
        elif env_method == 'tonnetz':
            print(bcolors.YELLOW + 'Using tonal centroid features for onset detection.' + bcolors.ENDC)
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr, hop_length=hop_length)
            o_env = librosa.onset.onset_strength(S=tonnetz,
                                                  sr=sr,
                                                  hop_length=hop_length,
                                                  lag=lag, max_size=max_size)
        elif env_method == 'pf':
            print(bcolors.YELLOW + 'Using poly features for onset detection.' + bcolors.ENDC)
            poly_features = librosa.feature.poly_features(y=y, sr=sr, hop_length=hop_length)
            o_env = librosa.onset.onset_strength(S=poly_features,
                                                  sr=sr,
                                                  hop_length=hop_length,
                                                  lag=lag, max_size=max_size)
        elif env_method == 'zcr':
            print(bcolors.YELLOW + 'Using zero-crossing rate for onset detection.' + bcolors.ENDC)
            zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=n_fft, hop_length=hop_length)
            o_env = librosa.onset.onset_strength(S=zcr,
                                                  sr=sr,
                                                  hop_length=hop_length,
                                                  lag=lag, max_size=max_size)
            
        onsets = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, hop_length=hop_length, backtrack=True, delta=self.args.onset_threshold)
        return onsets

    async def main(self):
        y, sr = librosa.load(self.args.input_file, sr=self.args.sample_rate)
        if sr != self.args.sample_rate:
            print(f'{bcolors.YELLOW}Loaded Sample rate is {sr}Hz!{bcolors.ENDC}')
            print(f'{bcolors.YELLOW}It was resampled from {self.args.sample_rate}Hz to {sr}Hz. Something could have gone wrong.{bcolors.ENDC}')
    
        if self.args.hop_size == 512:
            self.args.hop_size = int(librosa.time_to_samples(1./200, sr=sr))
        if self.args.source_separation is not None:
            # wait for the decomposition to finish
            from decomposer import Decomposer
            print(f'{bcolors.YELLOW}Decomposing the signal into harmonic and percussive components...{bcolors.ENDC}')
            decomposer = Decomposer(self.input_file, 'hpss', render=True, render_path=os.path.join(os.path.dirname(self.input_file), 'components'))
            H, P = await decomposer._decompose_hpss(y, n_fft=self.args.n_fft, hop_length=self.args.hop_size)
            if self.decompose == 'harmonic':
                y = H
            elif self.decompose == 'percussive':
                y = P
            else:
                raise ValueError('Invalid decomposition type.')
        
        onsets = self.compute_onsets(y, sr=self.args.sample_rate, n_fft=self.args.n_fft, hop_length=self.args.hop_size, fmin=self.args.fmin, fmax=self.args.fmax, env_method=self.args.onset_envelope)
        if len(onsets) == 0 or onsets is None:
            print(f'{bcolors.RED}No onsets detected! Try to reduce the onset threshold with -t.{bcolors.ENDC}')
            sys.exit(1)
            return
        ot = librosa.frames_to_time(onsets, sr=sr, hop_length=self.args.hop_size)
        onset_times=[]
        for i in range(len(ot)):
            onset_times.append(ot[i])
            
        total_length = round(librosa.get_duration(y=y, sr=sr),2)
        segs = onset_times + [total_length]
        
        segment_lengths = np.diff(segs)
        # segment_lengths
        
        print(f'\n{bcolors.GREEN}Detected {len(onsets)} onsets:{bcolors.ENDC}')
        print(f'{bcolors.CYAN}Total length: {total_length}s{bcolors.ENDC}')

        # collapse the list of onsets to only show the first 3 and last 3 onsets if there are too many
        if len(onsets) > 15:
            onsets_c = [onsets[0], onsets[1], onsets[2], onsets[-3], onsets[-2], onsets[-1]]
            onset_times_c = [onset_times[0], onset_times[1], onset_times[2], onset_times[-3], onset_times[-2], onset_times[-1]]
            segment_lengths_c = [segment_lengths[0], segment_lengths[1], segment_lengths[2], segment_lengths[-3], segment_lengths[-2], segment_lengths[-1]]
            # round the values to 3 decimal places
            onsets_c = [round(x, 3) for x in onsets_c]
            onset_times_c = [round(x, 3) for x in onset_times_c]
            segment_lengths_c = [round(x, 3) for x in segment_lengths_c]
            onsets_c.insert(3, '...')
            onset_times_c.insert(3, '...')
            segment_lengths_c.insert(3, '...')
        # if the number of onsets is less than 12, just show all of them
        # else show the first 3 and last 3 onsets with ... in between
        print(f'{bcolors.CYAN}Onsets Frames: {onsets_c if len(onsets) > 12 else onsets}{bcolors.ENDC}')
        print(f'{bcolors.CYAN}Onset times (sec): {onset_times_c if len(onsets) > 12 else onset_times}{bcolors.ENDC}')
        print(f'{bcolors.CYAN}Segment lengths (sec): {segment_lengths_c if len(onsets) > 12 else segment_lengths}{bcolors.ENDC}\n')
        return onsets



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segment an audio file based on its onsets and apply fade out to each segment')
    parser.add_argument("-i", "--input-file", type=str, help="Path to the audio file (wav, aif, aiff).", required=True)
    parser.add_argument('-o', '--output_directory', type=str, default=None, help='path to the output directory')
    parser.add_argument("-t", "--onset-threshold", type=float, default=0.1, help="Onset detection threshold. Default is 0.1.", required=False)
    parser.add_argument('-c', '--curve-type', type=str, default='exp', choices=['exp', 'log', 'linear', 's_curve', 'hann'], help='type of the fade out curve')
    parser.add_argument("-hpss", "--source-separation", type=str, default=None, choices=["harmonic", "percussive"], help="Decompose the signal into harmonic and percussive components, before computing segments.", required=False)
    parser.add_argument("--hop-size", type=int, default=512, help="Hop size. Default is 512.", required=False)
    parser.add_argument("--n-fft", type=int, default=2048, help="FFT size. Default is 2048.", required=False)
    parser.add_argument('-v', '--verbose', action='store_true', help='print verbose messages')

    args = parser.parse_args()

    if args.verbose:
        print(f'{bcolors.YELLOW}Verbose mode enabled.{bcolors.ENDC}')

    onset_detector = OnsetDetector(args)

    try:
        asyncio.run(onset_detector.main())
    except Exception as e:
        print(f'{bcolors.RED}Error: {type(e).__name__}! {e}{bcolors.ENDC}')
        traceback.print_exc()
        sys.exit(1)