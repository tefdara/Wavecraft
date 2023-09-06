import argparse
import librosa
import os
import sys
import soundfile as sf
import numpy as np
import traceback
import asyncio
from utils import apply_fadeout as fade, bcolors


class OnsetDetector:
    def __init__(self, input_file, peak_threshold=0.1, fade_duration=50, curve_type='exp', output_directory=None, verbose=False, decompose=None, min_length=0.1):
        self.input_file = input_file
        self.peak_threshold = peak_threshold
        self.fade_duration = fade_duration
        self.curve_type = curve_type
        self.output_directory = output_directory
        self.verbose = verbose
        self.decompose = decompose
        self.min_length = min_length

    def detect_onsets(self, y, sr, n_fft=1024, hop_length=512, n_mels=138, fmin=27.5, fmax=16000., lag=2, max_size=3):
        mel_spectogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
        o_env = librosa.onset.onset_strength(S=librosa.power_to_db(mel_spectogram, ref=np.max),
                                              sr=sr,
                                              hop_length=hop_length,
                                              lag=lag, max_size=max_size)
        onsets = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, hop_length=hop_length, delta=self.peak_threshold, backtrack=True)

        return onsets

    async def main(self):
        # Load an audio file
        y, sr = librosa.load(self.input_file, sr=None)
        n_fft = 1024
        hop_length = int(librosa.time_to_samples(1./200, sr=sr))
        lag = 2
        n_mels = 138
        fmin = 27.5
        fmax = 16000.
        max_size = 3

        if self.decompose:
            # wait for the decomposition to finish
            from decomposer import Decomposer
            print(f'{bcolors.YELLOW}Decomposing the signal into harmonic and percussive components...{bcolors.ENDC}')
            decomposer = Decomposer(self.input_file, 'hpss', render=True, render_path=os.path.join(os.path.dirname(self.input_file), 'components'))
            H, P = await decomposer._decompose_hpss(y, n_fft=n_fft, hop_length=hop_length)
            if self.decompose == 'harmonic':
                y = H
            elif self.decompose == 'percussive':
                y = P
            else:
                raise ValueError('Invalid decomposition type.')
            
        onsets = self.detect_onsets(y, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax, lag=lag, max_size=max_size)

        # Print the detected onsets (in samples)
        print(f'{bcolors.GREEN}Detected {len(onsets)} onsets{bcolors.ENDC}')

        # prompt user with miultiple choices, render the segments, export them as text file or exit
        user_input = input(f'Choose an action:\n1) Render segments\n2) Export segments as text file\n3) Exit\n')
        if user_input.lower() == '3':
            sys.exit()

        from segmentor import Segmentor
        
        segmentor = Segmentor(self.args)
        output_directory = os.path.join(os.path.dirname(self.input_file), 'segments')
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        if user_input.lower() == '1':
            segmentor.render_segments(y, sr, onsets, output_directory, self)
        elif user_input.lower() == '2':
            segmentor.save_segments_as_txt(onsets, output_directory, sr)

        print(f'{bcolors.GREEN}Done.{bcolors.ENDC}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segment an audio file based on its onsets and apply fade out to each segment')
    parser.add_argument('-i', '--input_file', type=str, help='path to the audio file', required=True)
    parser.add_argument('-t','--peak_threshold', type=float, default=0.1, help='peak threshold for onset detection')
    parser.add_argument('-f', '--fade_duration', type=int, default=50, help='duration of the fade out in milliseconds')
    parser.add_argument('-c', '--curve_type', type=str, default='exp', choices=['exp', 'log', 'linear', 's_curve', 'hann'], help='type of the fade out curve')
    parser.add_argument('-o', '--output_directory', type=str, default=None, help='path to the output directory')
    parser.add_argument('-v', '--verbose', action='store_true', help='print verbose messages')
    parser.add_argument('-d', '--decompose', type=str, default=None, choices=['harmonic', 'percussive'], help='decompose the signal into harmonic and percussive components')
    parser.add_argument('-l', '--min_length', type=float, default=0.1, help='minimum length of a segment in seconds')
    args = parser.parse_args()

    if args.verbose:
        print(f'{bcolors.YELLOW}Verbose mode enabled.{bcolors.ENDC}')

    onset_detector = OnsetDetector(args.input_file, args.peak_threshold, args.fade_duration, args.curve_type, args.output_directory, args.verbose, args.decompose, args.min_length)

    try:
        asyncio.run(onset_detector.main())
    except Exception as e:
        print(f'{bcolors.RED}Error: {type(e).__name__}! {e}{bcolors.ENDC}')
        traceback.print_exc()
        sys.exit(1)