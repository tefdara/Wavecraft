import argparse
import librosa
import os
import sys
import soundfile as sf
import numpy as np
import traceback
import asyncio
from utils import bcolors


class OnsetDetector:
    def __init__(self, args):
        self.args = args
        self.args.output_directory = self.args.output_directory or os.path.splitext(self.args.input_file)[0] + '_segments'
        
    def detect_onsets(self, y, sr, n_fft=1024, hop_length=512, n_mels=138, fmin=27.5, fmax=16000., lag=2, max_size=3):
        mel_spectogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
        o_env = librosa.onset.onset_strength(S=librosa.power_to_db(mel_spectogram, ref=np.max),
                                              sr=sr,
                                              hop_length=hop_length,
                                              lag=lag, max_size=max_size)
        onsets = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, hop_length=hop_length, delta=self.args.onset_threshold, backtrack=True)

        return onsets


    async def main(self):
        # Based on a paper by Boeck and Widmer, 2013 (Maximum filter vibrato suppression for onset detection)
        # Load an audio file
        y, sr = librosa.load(self.args.input_file, sr=self.args.sample_rate)
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
        
        onsets = self.detect_onsets(y, sr=self.args.sample_rate, n_fft=self.args.n_fft, hop_length=self.args.hop_size, fmin=self.args.fmin, fmax=self.args.fmax)
        onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=self.args.hop_size)
        segment_lengths = np.diff(onset_times)
        print(f'\n{bcolors.GREEN}Detected {len(onsets)} onsets:{bcolors.ENDC}')
        # collapse the list of onsets to only show the first 3 and last 3 onsets if there are too many
        if len(onsets) > 10:
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
        # if the number of onsets is less than 10, just show all of them
        # else show the first 3 and last 3 onsets with ... in between
        print(f'{bcolors.CYAN}Onsets Frames: {onsets_c if len(onsets) > 10 else onsets}{bcolors.ENDC}')
        print(f'{bcolors.CYAN}Onset times (sec): {onset_times_c if len(onsets) > 10 else onset_times}{bcolors.ENDC}')
        # Compute the segment lengths
        print(f'\n{bcolors.GREEN}Total of {len(segment_lengths)} segments{bcolors.ENDC}')
        print(f'{bcolors.CYAN}Segment lengths (sec): {segment_lengths_c if len(onsets) > 10 else segment_lengths}{bcolors.ENDC}')
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