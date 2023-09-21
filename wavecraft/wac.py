#!/usr/bin/env python3

import os, sys, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import librosa
from wavecraft import *

class WaveCraft:
    def __init__(self, args):
        self.args = args
        self.misc = self.args.operation in ["wmeta", "rmeta", "hpf", "info", "proxim"]
        
        self.files = []
        if self.args.input == '.':
            self.args.input = os.getcwd()
        # check if dir is home dir
        if self.args.input == os.path.expanduser('~'):
            print(f'\n{utils.bcolors.RED}You have selcted the home directory! Are you sure you want to go ahead?{utils.bcolors.ENDC}')
            user_input = input(f'\n1) Yes\n2) No\n')
            if user_input.lower() == '2':
                sys.exit(1)           
        if os.path.isdir(self.args.input):
            self.input_dir = self.args.input
            for file in os.listdir(self.args.input):
                if utils.check_format(file):
                    self.files.append(os.path.join(self.input_dir, file))
        # single file              
        else:
            if utils.check_format(self.args.input):
                self.files.append(self.args.input)
        if len(self.files) == 0:
            print(f'{utils.bcolors.RED}Could not find any valid files!{utils.bcolors.ENDC}')
            sys.exit()
                    
        if self.misc:
            return
        # store these as they will be adjusted for short signals
        self.n_fft = args.n_fft
        self.hop_size = args.hop_size
        self.window_length = 384 # for use with rythm features, otherwise window length = n_fft
        self.n_bins = 84
        self.n_mels = 128
        
    def main(self):
        
        if self.args.operation == "proxim":
            print(f'\n{utils.bcolors.GREEN}Finding {self.args.n_similar} similar sounds to {self.args.identifier or "all"}...{utils.bcolors.ENDC}\n')
            processor = proxi_metor(self.args)
            processor.main()
            return
        
        for file in self.files:
            self.args.input = file
            self.args.y=librosa.load(self.args.input, sr=self.args.sample_rate)[0]
            self.args.num_samples = len(self.args.y)
            self.args.duration = self.args.num_samples / self.args.sample_rate
            self.args.n_bins = 84
            self.args.num_frames = self.args.num_samples // self.args.hop_size
            self.args.window_length = 384
            self.args.n_fft, self.args.hop_size, self.args.window_length, self.args.n_bins, self.args.n_mels = utils.adjust_anal_res(self.args)
            
            if self.args.operation == "segment":
                print(f'\n{utils.bcolors.GREEN}Segmenting {os.path.basename(self.args.input)}...{utils.bcolors.ENDC}')
                processor = segmentor(self.args)
            elif self.args.operation == "extract":
                print(f'\n{utils.bcolors.GREEN}Extracting features from {os.path.basename(self.args.input)}...{utils.bcolors.ENDC}')
                processor = extractor(self.args)
            elif self.args.operation == "onset":
                print(f'\n{utils.bcolors.GREEN}Detecting onsets in {os.path.basename(self.args.input)}...{utils.bcolors.ENDC}')
                processor = onset_detector(self.args)
            elif self.args.operation == "beat":
                print(f'\n{utils.bcolors.GREEN}Detecting beats in {os.path.basename(self.args.input)}...{utils.bcolors.ENDC}')
                processor = beat_detector(self.args)
            elif self.args.operation == "decomp":
                print(f'\n{utils.bcolors.GREEN}Decomposing {os.path.basename(self.args.input)}...{utils.bcolors.ENDC}')
                processor = decomposer(self.args)
            else:
                processor = utils
  
            if self.misc == False:
                processor.main()
            else:
                if self.args.operation == "wmeta":
                    print(f'\n{utils.bcolors.YELLOW}Writing metadata to {os.path.basename(self.args.input)}...{utils.bcolors.ENDC}')
                    if(self.args.meta_file):
                        self.args.meta = utils.load_json(self.args.meta_file)
                    processor.write_metadata(self.args.input, self.args.meta)
                if self.args.operation == "rmeta":
                    print(f'\n{utils.bcolors.YELLOW}Extracting metadata from {os.path.basename(self.args.input)}...{utils.bcolors.ENDC}')
                    processor.extract_metadata(self.args.input, self.args)
                if self.args.operation == "hpf":
                    print(f'\n{utils.bcolors.YELLOW}Applying high-pass filter to {os.path.basename(self.args.input)}...{utils.bcolors.ENDC}')
                    processor.filter(self.args.input, self.args.sample_rate, self.args.filter_frequency, btype=self.args.filter_type)
            
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

    parser.add_argument("operation", type=str, choices=["segment", "extract", "proxim", "onset", "beat", "decomp", "filter", "norm", "wmeta", "rmeta", "info"], 
                    help="Operation to perform. Choices: segment:segmentaion, extract:feature extraction, proxim: proximity learning, onset:onset detection, beat:beat detection, decomp:decomposition, filter:filter, norm:normalise, wmeta:write metadata, rmeta:read metadata, info:file info",
                    metavar='operation',
                    nargs='?') 
    
    # Create a group for input arguments
    input_group = parser.add_argument_group('Input')
    input_group.add_argument("-i", "--input", type=str, help="Path to the audio, metadata or dataset file. It can be a directory for batch processing. It is valid for all operations", required=True)
    input_group.add_argument("-it", "--input-text", type=str, required=False,
                        help="The text file containing the segmentation data. Defaults to the nameofaudio.txt")

    # Create a group for output arguments
    output_group = parser.add_argument_group('Output')
    output_group.add_argument("-o", "--output-directory", type=str, default=None, help="Path to the output directory. Optional.", required=False)
    output_group.add_argument("-st", "--save-txt", action='store_true', help="Save segment times to a text file.")
    
    # Audio settings
    audio_settings_group = parser.add_argument_group('Audio settings')
    audio_settings_group.add_argument("-sr","--sample-rate", type=int, default=44100, help="Sample rate of the audio file. Default is 44100.", required=False)
    audio_settings_group.add_argument("--fmin", type=float, default=30, help="Minimum frequency. Default is 30.", required=False)
    audio_settings_group.add_argument("--fmax", type=float, default=16000, help="Maximum frequency. Default is 16000", required=False)
    audio_settings_group.add_argument("--n-fft", type=int, default=2048, help="FFT size. Default is 2048.", required=False)
    audio_settings_group.add_argument("--hop-size", type=int, default=512, help="Hop size. Default is 512.", required=False)

    # Create a group for segmentation arguments
    segmentation_group = parser.add_argument_group(title='Segmentation -> segment', description='splits the audio file into segments based on the provided arguments')
    segmentation_group.add_argument("-m", "--segmentation-method", type=str, choices=["onset", "beat", "text"], required=False, default="onset",
                        help="Segmentation method to use.")
    segmentation_group.add_argument("-ml", "--min-length", type=float, default=0.1, help="Minimum length of a segment in seconds. Default is 0.1s.\
                        anything shorter won't be used", required=False)
    segmentation_group.add_argument("-f", "--fade-duration", type=int, default=20, help="Duration in ms for fade in and out. Default is 50ms.", required=False)
    segmentation_group.add_argument("-c", "--curve-type", type=str, choices=['exp', 'log', 'linear', 's_curve','hann'], default="exp",\
                        help="Type of curve to use for fade in and out. Default is exponential.", required=False)
    segmentation_group.add_argument("-t", "--onset-threshold", type=float, default=0.1, help="Onset detection threshold. Default is 0.1.", required=False)
    segmentation_group.add_argument("-ts", "--trim-silence", type=float, default=60, help="Trim silence from the beginning and end of the audio file. Default is 60 db.", required=False)
    segmentation_group.add_argument("-oe", "--onset-envelope", type=str, choices=['mel', 'cqt', 'stft', 'cqt_chr', 'mfcc', 'rms', 'zcr', 'cens', 'tmpg', 'ftmpg', 'tonnetz', 'pf'], default="mel",\
                        help="Onset envelope to use for onset detection. Default is mel (mel spectrogram). Choices are: mel (mel spectrogram), cqt (constant-Q transform), stft (short-time Fourier transform), cqt_chr (chroma constant-Q transform), mfcc (Mel-frequency cepstral coefficients), rms (root-mean-square energy), zcr (zero-crossing rate), cens (chroma energy normalized statistics), tmpg (tempogram), ftmpg (fourier tempogram), tonnetz (tonal centroid features), pf (poly features).", required=False)
    
    # Feature extraction arguments
    feature_extraction_group = parser.add_argument_group(title='Feature extraction -> extract', description='extracts features from the audio file')
    feature_extraction_group.add_argument("-fex", "--feature-extractor", type=str, choices=['mel', 'cqt', 'stft', 'cqt_chr', 'mfcc', 'rms', 'zcr', 'cens', 'tmpg', 'ftmpg', 'tonnetz', 'pf'], default=None,\
                                        help="Feature extractor to use. Default is all. Choices are: mel (mel spectrogram), cqt (constant-Q transform), stft (short-time Fourier transform), cqt_chr (chroma constant-Q transform), mfcc (Mel-frequency cepstral coefficients), rms (root-mean-square energy), zcr (zero-crossing rate), cens (chroma energy normalized statistics), tmpg (tempogram), ftmpg (fourier tempogram), tonnetz (tonal centroid features), pf (poly features).", required=False)
    feature_extraction_group.add_argument("-fdic", "--flatten-dictionary", type=str, default='False', help="Flattens the output dictionary , default is True", required=False, 
                                          choices=['True', 'False'], nargs='?')
    
    # Proximity metric calculation arguments
    proxi_metor_group = parser.add_argument_group(title='Proximity metric calculation -> proxim', description= 'finds the most similar sounds based on the features dataset')
    proxi_metor_group.add_argument('-ns', '--n-similar', type=int, default=5,
                        help='Number of similar sounds to retrieve')
    proxi_metor_group.add_argument('-id', '--identifier', type=str, required=False,
                        help='Identifier to test, i.e., the name of sound file. If not provided, all sounds in the dataset will be tested against each other')
    proxi_metor_group.add_argument('-cls', '--class_to_analyse', type=str, default='stats',
                        help='Class to analyse. Default: stats. If not provided, all classes will be analysed. Note that this option can produce unexpected results if the dataset contains multiple classes with different dimensions')
    proxi_metor_group.add_argument('-mt', '--metric-to-analyze', type=str, default=None,
                        help='Metric to analyze')
    proxi_metor_group.add_argument('-ops', action='store_true', default=False,
                        help='Use opetions file to fine tune the metric learning')
    proxi_metor_group.add_argument('-mn', '--n-max', type=int, default=-1, 
                        help='Max number of similar files to retrieve, Default: -1 (all)')
    
    # Decomposition arguments
    decomposition_group = parser.add_argument_group(title='Decomposition -> decomp', description='decomposes the audio file into harmonic, percussive or n components')
    decomposition_group.add_argument("-n", "--n-components", type=int, default=4, help="Number of components to use for decomposition.", required=False)
    decomposition_group.add_argument("-hpss", "--source-separation", type=str, default=None, choices=["harmonic", "percussive"], help="Decompose the signal into harmonic and percussive components, before computing segments.", required=False)
    
    # Beat detection arguments
    beat_group = parser.add_argument_group(title='Beat detection -> beat', description='detects beats in the audio file')
    beat_group.add_argument("-k", type=int, default=5, help="Number of beat clusters to detect. Default is 5.", required=False)
    
    # Filter arguments
    filter_group = parser.add_argument_group(title='Filter -> filter', description='applies a high / low pass filter to the audio file')
    filter_group.add_argument("-ff", "--filter-frequency", type=int, default=40, 
                        help="Frequency to use for the high-pass filter. Default is 40 Hz. Set to 0 to disable", required=False)
    filter_group.add_argument("-ft", "--filter-type", type=str, choices=['high', 'low'], default="high", 
                        help="Type of filter to use. Default is high-pass.", required=False)

    # Normalization arguments
    normalization_group = parser.add_argument_group(title='Normalization -> norm', description='normalizes the audio file')
    normalization_group.add_argument("-nl", "--normalisation-level", type=float, default=-3, required=False,
                        help="Normalisation level, default is -3 db.")
    normalization_group.add_argument("-nm", "--normalisation-mode", type=str, default="peak", choices=["peak", "rms", "loudness"], 
                        help="Normalisation mode; default is RMS.", required=False)
    
    # Metadata arguments
    metadata_group = parser.add_argument_group(title='Metadata -> wmeta | rmeta', description='write metadata to the audio file | read metadata from the audio file')
    metadata_group.add_argument("--meta", type=str, help="List of metadata or comments to write to the file. Default is None.", required=False, nargs='+')
    metadata_group.add_argument("-mf", "--meta-file", type=str, help="Path to a JSON metadata file. Default is None.", required=False)

    args = parser.parse_args()

    wave_craft = WaveCraft(args)
    wave_craft.main()
    
    