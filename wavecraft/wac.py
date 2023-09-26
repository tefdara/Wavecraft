#!/usr/bin/env python3

import os, sys, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import librosa, soundfile as sf
from wavecraft import *



def main(args):
    dsp = args.operation not in ["wmeta", "rmeta", "info"]
    process = args.operation in ["filter", "norm"]
    # store these as they will be adjusted for short signals
    n_fft = args.n_fft
    hop_size = args.hop_size
    window_length = args.window_length = 384 # for use with rythm features, otherwise window length = n_fft
    n_bins = args.n_bins = 84
    n_mels = args.n_mels = 128
    files = load_files(args.input)
    print('')
    
    if args.operation == "proxim":
        extra = utils.extra_log_string('Calculating', '')
        utils.info_logger.info('proximity metric', extra=extra)
        craft = ProxiMetor(args)
        craft.main()
        return
    
    if process:
        processor = Processor(args, mode='render')

    for file in files:
        args.input = file
        if dsp:
            try:
                if process:
                    args.y, args.sample_rate = sf.read(file, dtype='float32')
                    args.output = input
                else:
                    args.y=librosa.load(file, sr=args.sample_rate)[0]
            except RuntimeError:
                utils.error_logger.error(f'Could not load {file}!')
                continue
            if not librosa.util.valid_audio(args.y):
                utils.error_logger.error(f'{file} is not a valid audio file!')
        
        args.num_samples = args.y.shape[-1]
        args.duration = args.num_samples / args.sample_rate
        args.n_fft, args.hop_size, args.window_length, args.n_bins, args.n_mels = utils.adjust_anal_res(args)
        args.num_frames = int(args.num_samples / args.hop_size)
        extra = utils.extra_log_string('', f'{os.path.basename(file)}')
        if args.operation == "segment":
            utils.info_logger.info('Segmenting', extra=extra)
            craft = Segmentor(args)
            craft.main()
        elif args.operation == "extract":
            utils.info_logger.info('Extracting features for', extra=extra)
            craft = Extractor(args)
            craft.main()
        elif args.operation == "onset":
            utils.info_logger.info('Detecting onsets for', extra=extra)
            craft = OnsetDetector(args)
            craft.main()
        elif args.operation == "beat":
            utils.info_logger.info('Detecting beats for', extra=extra)
            craft = BeatDetector(args)
            craft.main()
        elif args.operation == "decomp":
            utils.info_logger.info('Decomposing', extra=extra)
            craft = Decomposer(args)
            craft.main()
        elif args.operation == "filter":
            utils.info_logger.info('Applying filter', extra=extra)
            processor.filter(args.y, args.sample_rate, args.filter_frequency, args.filter_type)
        elif args.operation == "norm":
            utils.info_logger.info('Normalising', extra=extra)
            processor.normalise_audio(args.y, args.sample_rate, args.normalisation_level, args.normalisation_mode)                    
            
        else:
            if args.operation == "wmeta":
                utils.info_logger.info('Writing metadata', extra=extra)
                if(args.meta_file):
                    args.meta = utils.load_json(args.meta_file)
                else:
                    utils.error_logger.error('No metadata file provided!')
                    sys.exit()
                utils.write_metadata(file, args.meta)
            if args.operation == "rmeta":
                utils.info_logger.info('Extracting metadata', extra=extra)
                utils.extract_metadata(file, args)
            
        args.n_fft = n_fft
        args.hop_size = hop_size
        args.window_length = window_length
        args.n_bins = n_bins
        args.n_mels = n_mels 

 
def load_files(input):
    files = []
    if input == None or input == '':
        print(f'{utils.bcolors.RED}No input provided!{utils.bcolors.ENDC}')
        sys.exit()
    if input == '.':
        input = os.getcwd()
    # check if dir is home dir
    if input == os.path.expanduser('~'):
        print(f'\n{utils.bcolors.RED}You have selcted the home directory! Are you sure you want to go ahead?{utils.bcolors.ENDC}')
        user_input = input(f'\n1) Yes\n2) No\n')
        if user_input.lower() == '2':
            sys.exit(1)           
    if os.path.isdir(input):
        input_dir = input
        for file in os.listdir(input):
            if utils.check_format(file):
                files.append(os.path.join(input_dir, file))
    # single file              
    else:
        if utils.check_format(input):
            files.append(input)
    if len(files) == 0:
        print(f'{utils.bcolors.RED}Could not find any valid files!{utils.bcolors.ENDC}')
        sys.exit()
    return files




if __name__ == "__main__":
    
    usage = "usage: wave_craft.py operation [options] arg"
    help = "wave_craft.py -h, --help for more details"
    parser = argparse.ArgumentParser(prog='Wave Craft', description="Split audio files based on segments from a text file.", 
                                     usage=usage +'\n'+help, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("operation", type=str, choices=["segment", "extract", "proxim", "onset", "beat", "decomp", "filter", "norm", "fade", "wmeta", "rmeta", "info"], 
                    help="Operation to perform. Choices: segment:segmentaion, extract:feature extraction, proxim: proximity learning, onset:onset detection, beat:beat detection, decomp:decomposition, filter:filter, norm:normalisation, fade:fade, wmeta:write metadata, rmeta:read metadata, info:file info",
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
    audio_settings_group = parser.add_argument_group('Audio settings', description='These settings apply to all operations where relevant')
    audio_settings_group.add_argument("-sr","--sample-rate", type=int, default=48000, help="Sample rate of the audio file. Default is 44100.", required=False)
    audio_settings_group.add_argument("--fmin", type=float, default=30, help="Minimum analysis frequency. Default is 30.", required=False)
    audio_settings_group.add_argument("--fmax", type=float, default=16000, help="Maximum analysis frequency. Default is 16000", required=False)
    audio_settings_group.add_argument("--n-fft", type=int, default=2048, help="FFT size. Default is 2048.", required=False)
    audio_settings_group.add_argument("--hop-size", type=int, default=512, help="Hop size. Default is 512.", required=False)
    audio_settings_group.add_argument("-fi", "--fade-in", type=int, default=50, help="Duration in ms for fade in. Default is 50ms.", required=False)
    audio_settings_group.add_argument("-fo", "--fade-out", type=int, default=100, help="Duration in ms for fade in. Default is 100ms.", required=False)
    audio_settings_group.add_argument("-c", "--curve-type", type=str, choices=['exp', 'log', 'linear', 's_curve','hann'], default="exp",\
                        help="Type of curve to use for fade in and out. Default is exponential.", required=False)

    # Create a group for segmentation arguments
    segmentation_group = parser.add_argument_group(title='Segmentation -> segment', description='splits the audio file into segments based on the provided arguments')
    segmentation_group.add_argument("-m", "--segmentation-method", type=str, choices=["onset", "beat", "text"], required=False, default="onset",
                        help="Segmentation method to use.")
    segmentation_group.add_argument("-ml", "--min-length", type=float, default=0.1, help="Minimum length of a segment in seconds. Default is 0.1s.\
                        anything shorter won't be used", required=False)
    segmentation_group.add_argument("-t", "--onset-threshold", type=float, default=0.08, help="Onset detection threshold. Default is 0.08.", required=False)
    segmentation_group.add_argument("-ts", "--trim-silence", type=float, default=-65, help="Trim silence from the beginning and end of the audio file. Default is -60 db.", required=False)
    segmentation_group.add_argument("-oe", "--onset-envelope", type=str, choices=['mel', 'mfcc', 'cqt_chr', 'rms', 'zcr', 'cens', 'tmpg', 'ftmpg', 'tonnetz'], default="mel",\
                        help="Onset envelope to use for onset detection. Default is mel (mel spectrogram).\n Choices are: mel (mel spectrogram), mfcc (Mel-frequency cepstral coefficients), cqt_chr (chroma constant-Q transform), rms (root-mean-square energy), zcr (zero-crossing rate), cens (chroma energy normalized statistics), tmpg (tempogram), ftmpg (fourier tempogram), tonnetz (tonal centroid features)", required=False)
    segmentation_group.add_argument("-bl", "--backtrack-length", type=float, default=20, help="Backtrack length in miliseconds. Backtracks the segments from the detected onsets. Default is 20ms.", required=False)
                                    
    # Feature extraction arguments
    feature_extraction_group = parser.add_argument_group(title='Feature extraction -> extract', description='extracts features from the audio file')
    feature_extraction_group.add_argument("-fex", "--feature-extractor", type=str, choices=['mel', 'cqt', 'stft', 'cqt_chr', 'mfcc', 'rms', 'zcr', 'cens', 'tmpg', 'ftmpg', 'tonnetz', 'pf'], default=None,\
                                        help="Feature extractor to use. Default is all. Choices are: mel (mel spectrogram), cqt (constant-Q transform), stft (short-time Fourier transform), cqt_chr (chroma constant-Q transform), mfcc (Mel-frequency cepstral coefficients), rms (root-mean-square energy), zcr (zero-crossing rate), cens (chroma energy normalized statistics), tmpg (tempogram), ftmpg (fourier tempogram), tonnetz (tonal centroid features), pf (poly features).", required=False)
    feature_extraction_group.add_argument("-fdic", "--flatten-dictionary", action='store_true', default=False, help="Flatten the dictionary of features. Default is False.", required=False)
    
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
    main(args)
    
    