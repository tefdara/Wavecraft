#!/usr/bin/env python3

import os, sys, argparse
import librosa
import utils
from segmentor import Segmentor
from onset_detector import OnsetDetector
from beat_detector import BeatDetector
from decomposer import Decomposer
from feature_extractor import Extractor

class WaveCraft:
    def __init__(self, args):
        self.args = args
        self.misc = False
        # store these as they will be adjusted for short signals
        self.n_fft = args.n_fft
        self.hop_size = args.hop_size
        self.window_length = 384 # for use with rythm features, otherwise window length = n_fft
        self.n_bins = 84
        self.n_mels = 128
        
        self.files = []           
        if os.path.isdir(self.args.input_file):
            self.input_dir = self.args.input_file
            for file in os.listdir(self.args.input_file):
                if utils.check_format(file):
                    self.files.append(os.path.join(self.input_dir, file))
        # single file              
        else:
            if utils.check_format(self.args.input_file):
                self.files.append(self.args.input_file)
        if len(self.files) == 0:
            print(f'{utils.bcolors.RED}Could not find any valid files!{utils.bcolors.ENDC}')
            sys.exit()
        
    def main(self):
        
        for file in self.files:
            self.args.input_file = file
            self.args.y=librosa.load(self.args.input_file, sr=self.args.sample_rate)[0]
            self.args.duration=len(self.args.y)/self.args.sample_rate
            self.args.num_samples = len(self.args.y)
            self.args.n_bins = 84
            self.args.num_frames = self.args.num_samples // self.args.hop_size
            self.args.window_length = 384
            self.args.n_fft, self.args.hop_size, self.args.window_length, self.args.n_bins, self.args.n_mels = utils.adjust_anal_res(self.args)
            
            if self.args.operation == "segment":
                print(f'\n{utils.bcolors.MAGENTA}Segmenting {os.path.basename(self.args.input_file)}...{utils.bcolors.ENDC}')
                processor = Segmentor(self.args)
            elif self.args.operation == "extract":
                print(f'\n{utils.bcolors.YELLOW}Extracting features from {os.path.basename(self.args.input_file)}...{utils.bcolors.ENDC}')
                processor = Extractor(self.args)
            elif self.args.operation == "onset":
                print(f'\n{utils.bcolors.YELLOW}Detecting onsets in {os.path.basename(self.args.input_file)}...{utils.bcolors.ENDC}')
                processor = OnsetDetector(self.args)
            elif self.args.operation == "beat":
                print(f'\n{utils.bcolors.YELLOW}Detecting beats in {os.path.basename(self.args.input_file)}...{utils.bcolors.ENDC}')
                processor = BeatDetector(self.args)
            elif self.args.operation == "decomp":
                print(f'\n{utils.bcolors.YELLOW}Decomposing {os.path.basename(self.args.input_file)}...{utils.bcolors.ENDC}')
                processor = Decomposer(self.args)
            else:
                processor = utils
                self.misc = True
  
            if self.misc == False:
                processor.main()
            else:
                if self.args.operation == "wmeta":
                    print(f'\n{utils.bcolors.YELLOW}Writing metadata to {os.path.basename(self.args.input_file)}...{utils.bcolors.ENDC}')
                    if(self.args.meta_file):
                        self.args.meta = utils.load_json(self.args.meta_file)
                    processor.write_metadata(self.args.input_file, self.args.meta)
                if self.args.operation == "rmeta":
                    print(f'\n{utils.bcolors.YELLOW}Extracting metadata from {os.path.basename(self.args.input_file)}...{utils.bcolors.ENDC}')
                    processor.extract_metadata(self.args.input_file, self.args)
                if self.args.operation == "hpf":
                    print(f'\n{utils.bcolors.YELLOW}Applying high-pass filter to {os.path.basename(self.args.input_file)}...{utils.bcolors.ENDC}')
                    processor.filter(self.args.input_file, self.args.sample_rate, self.args.filter_frequency, btype=self.args.filter_type)
            
            self.args.n_fft = self.n_fft
            self.args.hop_size = self.hop_size
            self.args.window_length = self.window_length
            self.args.n_bins = self.n_bins
            self.args.n_mels = self.n_mels       





if __name__ == "__main__":
    usage = "usage: wave_craft.py operation [options] arg"
    help = "wave_craft.py -h, --help for more details"
    parser = argparse.ArgumentParser(prog='Wave Craft', description="Split audio files based on segments from a text file.", 
                                     usage=usage +'\n'+help, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("operation", type=str, choices=["segment", "extract", "onset", "beat", "decomp", "hpf", "wmeta", "rmeta", "info"], 
                    help="Operation to perform. Choices: segment:segmentaion, extract:feature extraction, onset:onset detection, beat:beat detection, decomp:decomposition, hpf:high-pass filter, wmeta:write metadata, rmeta:read metadata, info:file info",
                    metavar='operation',
                    nargs='?') 
    
    # Create a group for input arguments
    input_group = parser.add_argument_group('Input')
    input_group.add_argument("-i", "--input-file", type=str, help="Path to the audio file or a directory for batch processing.", required=True)
    input_group.add_argument("-it", "--input-text", type=str, required=False,
                        help="The text file containing the segmentation data. Defaults to the nameofaudio.txt")

    # Create a group for output arguments
    output_group = parser.add_argument_group('Output')
    output_group.add_argument("-o", "--output-directory", type=str, default=None, help="Path to the output directory. Optional.", required=False)
    output_group.add_argument("-st", "--save-txt", action='store_true', help="Save segment times to a text file.")

    # Create a group for segmentation arguments
    segmentation_group = parser.add_argument_group('Segmentation')
    segmentation_group.add_argument("-m", "--segmentation-method", type=str, choices=["onset", "beat", "text"], required=False, default="onset",
                        help="Segmentation method to use.")
    segmentation_group.add_argument("-ml", "--min-length", type=float, default=0.1, help="Minimum length of a segment in seconds. Default is 0.1s.\
                        anything shorter won't be used", required=False)
    segmentation_group.add_argument("-f", "--fade-duration", type=int, default=20, help="Duration in ms for fade in and out. Default is 50ms.", required=False)
    segmentation_group.add_argument("-c", "--curve-type", type=str, choices=['exp', 'log', 'linear', 's_curve','hann'], default="exp",\
                        help="Type of curve to use for fade in and out. Default is exponential.", required=False)
    segmentation_group.add_argument("-t", "--onset-threshold", type=float, default=0.1, help="Onset detection threshold. Default is 0.1.", required=False)
    # trim silence max db 
    segmentation_group.add_argument("-ts", "--trim-silence", type=float, default=60, help="Trim silence from the beginning and end of the audio file. Default is 60 db.", required=False)
    segmentation_group.add_argument("-oe", "--onset-envelope", type=str, choices=['mel', 'cqt', 'stft', 'cqt_chr', 'mfcc', 'rms', 'zcr', 'cens', 'tmpg', 'ftmpg', 'tonnetz', 'pf'], default="mel",\
                        help="Onset envelope to use for onset detection. Default is mel (mel spectrogram). Choices are: mel (mel spectrogram), cqt (constant-Q transform), stft (short-time Fourier transform), cqt_chr (chroma constant-Q transform), mfcc (Mel-frequency cepstral coefficients), rms (root-mean-square energy), zcr (zero-crossing rate), cens (chroma energy normalized statistics), tmpg (tempogram), ftmpg (fourier tempogram), tonnetz (tonal centroid features), pf (poly features).", required=False)
    # Create a group for decomposition arguments
    decomposition_group = parser.add_argument_group('Decomposition')
    decomposition_group.add_argument("-n", "--n-components", type=int, default=4, help="Number of components to use for decomposition.", required=False)
    decomposition_group.add_argument("-hpss", "--source-separation", type=str, default=None, choices=["harmonic", "percussive"], help="Decompose the signal into harmonic and percussive components, before computing segments.", required=False)
    
    # Create a group for beat detection arguments
    beat_group = parser.add_argument_group('Beat detection')
    beat_group.add_argument("-k", type=int, default=5, help="Number of beat clusters to detect. Default is 5.", required=False)
    
    # Create a group for filter arguments
    filter_group = parser.add_argument_group('Filter')
    filter_group.add_argument("-ff", "--filter-frequency", type=int, default=40, 
                        help="Frequency to use for the high-pass filter. Default is 40 Hz. Set to 0 to disable", required=False)
    filter_group.add_argument("-ft", "--filter-type", type=str, choices=['high', 'low'], default="high", 
                        help="Type of filter to use. Default is high-pass.", required=False)

    # Create a group for normalization arguments
    normalization_group = parser.add_argument_group('Normalization')
    normalization_group.add_argument("-nl", "--normalisation-level", type=float, default=-3, required=False,
                        help="Normalisation level, default is -3 db.")
    normalization_group.add_argument("-nm", "--normalisation-mode", type=str, default="peak", choices=["peak", "rms", "loudness"], 
                        help="Normalisation mode; default is RMS.", required=False)
    
    # Create group for metadata arguments
    metadata_group = parser.add_argument_group('Metadata')
    metadata_group.add_argument("--meta", type=str, help="List of metadata or comments to write to the file. Default is None.", required=False, nargs='+')
    metadata_group.add_argument("-mf", "--meta-file", type=str, help="Path to a JSON metadata file. Default is None.", required=False)

    # Create a group for other arguments
    other_group = parser.add_argument_group('Audio settings')
    other_group.add_argument("-sr","--sample-rate", type=int, default=48000, help="Sample rate of the audio file. Default is 48000.", required=False)
    other_group.add_argument("--fmin", type=float, default=20, help="Minimum frequency. Default is 27.5.", required=False)
    other_group.add_argument("--fmax", type=float, default=16000, help="Maximum frequency. Default is 16000", required=False)
    other_group.add_argument("--n-fft", type=int, default=2048, help="FFT size. Default is 2048.", required=False)
    other_group.add_argument("--hop-size", type=int, default=512, help="Hop size. Default is 512.", required=False)

    args = parser.parse_args()

    wave_craft = WaveCraft(args)
    wave_craft.main()
    
    