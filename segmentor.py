#!/usr/bin/env python3.9
import os
import librosa
import argparse
import soundfile as sf
from utils import bcolors, apply_fadeout as fade
from onset_detector import OnsetDetector
from beat_detector import BeatDetector

class Segmentor:
    def __init__(self, args):
        self.args = args
        self.y, _ = librosa.load(self.args.input_file, sr=self.args.sample_rate)
        self.output_directory = self.args.output_directory or os.path.splitext(self.args.input_file)[0] + '_segments'

    def render_segments(self, y, sr, onsets, output_directory):
        for i in range(len(onsets) - 1):
            start_sample = onsets[i]
            end_sample = onsets[i + 1]
            segment = y[start_sample:end_sample]
            # skip segments that are too short
            segment_length = len(segment) / sr
            if segment_length < self.args.min_length:
                print(f'{bcolors.YELLOW}Skipping segment {i} because it\'s too short.{bcolors.ENDC}')
                continue
            faded_seg = fade(segment, sr, fade_duration=self.args.fade_duration, curve_type=self.args.curve_type)
            segment_path = os.path.join(output_directory, os.path.basename(self.args.input_file).split('.')[0]+f'_{i}.wav')
            print(f'{bcolors.GREEN}Saving segment {i} to {segment_path}.{bcolors.ENDC}')
            # Save segment to a new audio file
            sf.write(segment_path, faded_seg, sr, format='WAV', subtype='FLOAT')

    def save_segments_as_txt(self, onsets, output_directory, sr):
        with open(os.path.join(output_directory, 'segments.txt'), 'w') as file:
            for i in range(len(onsets) - 1):
                start_sample = onsets[i]
                end_sample = onsets[i + 1]
                start_time = start_sample / sr
                end_time = end_sample / sr
                file.write(f'{start_time} {end_time}\n')

    def segment_using_txt(self, audio_path, txt_path, output_folder, file_format):
        
        y, sr = librosa.load(audio_path, sr=None)

        # Read the text file and split the audio based on the segments provided
        with open(txt_path, 'r') as file:
            lines = file.readlines()

            for i, line in enumerate(lines):
                tokens = line.split()
                start_time, end_time = tokens[:2]
                start_time = float(start_time) * 1000  # convert to milliseconds
                end_time = float(end_time) * 1000    # convert to milliseconds

                start_sample = int(start_time * sr / 1000)
                end_sample = int(end_time * sr / 1000)
                segment = y[start_sample:end_sample]

                # Adding fade-in and fade-out effects
                segment = fade(segment, sr, fade_duration=self.args.fade_duration)

                segment_path = os.path.join(output_folder, f"segment_{i}.wav")
                sf.write(segment_path, segment, sr, format='WAV', subtype='FLOAT')
                
    def segment_using_boundaries(self, boundaries):
        segments = []
        for i in range(len(boundaries) - 1):
            start_sample = boundaries[i]
            end_sample = boundaries[i + 1]
            segment = self.y[start_sample:end_sample]
            segments.append(segment)
        return segments
    
    def main(self):
        if self.args.segmentation_method == 'onset':
            detector = OnsetDetector(self.args)
            segments = detector.detect_onsets()
        elif self.args.segmentation_method == 'beat':
            detector = BeatDetector(self.args)
            segments = self.segment_using_boundaries(detector.detect_beats())

        if self.args.segmentation_method == 'text':
            if(not self.args.input_text):
                self.args.input_text = os.path.splitext(self.args.input_file)[0] + '.txt'
            self.segment_using_txt(self.args.input_file, self.args.input_text, self.args.output_directory, self.args.file_format)
        # else:
        #     os.makedirs(self.output_directory, exist_ok=True)
        #     self.render_segments(self.y, self.args.sample_rate, segments, self.output_directory)

        if self.args.save_txt:
            self.save_segments_as_txt(segments, self.output_directory, self.args.sr)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split audio files based on segments from a text file.")
    parser.add_argument("-i", "--input-file", type=str, help="Path to the audio file (wav, aif, aiff).", required=True)
    parser.add_argument("-tx","--input-text", type=str, help="The text file containing the segmentation data. Defaults to the nameofaudio.txt", required=False)
    parser.add_argument("-o", "--output-directory", type=str, default=None, help="Path to the output directory. Optional.", required=False)
    parser.add_argument("-m","--segmentation-method", type=str, choices=["onset", "beat", "text"], help="Segmentation method to use.", required=True)
    parser.add_argument("--save-txt", action='store_true', help="Save segment times to a text file.")
    parser.add_argument("--min-length", type=float, default=0.1, help="Minimum length of a segment in seconds. Default is 0.1s.\
                        anything shorter won't be used", required=False)
    parser.add_argument("--fade-duration", type=int, default=50, help="Duration in ms for fade in and out. Default is 50ms.", required=False)
    parser.add_argument("--curve-type", type=str, choices=['exp', 'log', 'linear', 's_curve','hann'], default="exp",\
                        help="Type of curve to use for fade in and out. Default is exponential.", required=False)
    
    parser.add_argument("--sample-rate", type=int, default=48000, help="Sample rate of the audio file. Default is 48000.", required=False)
    parser.add_argument("--fmin", type=float, default=27.5, help="Minimum frequency. Default is 27.5.", required=False)
    parser.add_argument("--fmax", type=float, default=16000, help="Maximum frequency. Optional.", required=False)
    parser.add_argument("--onset-threshold", type=float, default=0.1, help="Onset detection threshold. Default is 0.5.", required=False)
    parser.add_argument("--onset-min-distance", type=float, default=0.0, help="Minimum distance between onsets in seconds. Default is 0.0.", required=False)
    parser.add_argument("--decompose", type=str, choices=["harmonic", "percussive"], help="Decompose the signal into harmonic and percussive components.", required=False)
    parser.add_argument("--n-fft", type=int, default=2048, help="FFT size. Default is 2048.", required=False)
    parser.add_argument("--hop-size", type=int, default=512, help="Hop size. Default is 512.", required=False)
    parser.add_argument("-k", type=int, default=5, help="Number of beat clusters. Default is 5.", required=False)

    # text segmentation parameters
    # parser.add_argument("--segment-times", type=str, default=None, help="Segment times in seconds separated by commas. Optional.", required=False)

    args = parser.parse_args()

    segmentor = Segmentor(args)
    segmentor.main()
    
    