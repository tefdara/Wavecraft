#!/usr/bin/env python3

import os, sys, argparse, time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import librosa, soundfile as sf
from wavecraft import * 
from wavecraft.debug import Debug as debug


def main(args, revert=None):
    t=0
    while t <= 1.01:
        utils.progress_bar(t, 1, message=args.operation)
        t+=0.01
        time.sleep(0.005)
    
    if args.operation == "proxim":
        debug.log_info('Calculating <proximity metric>')
        craft = ProxiMetor(args)
        craft.main()
        return
    
    dsp = args.operation not in ["wmeta", "info"]
    process = args.operation not in ["segment", "extract", "proxim", "onset", "beat", "decomp"]
    # store these as they will be adjusted for short signals
    n_fft = args.n_fft
    hop_size = args.hop_size
    window_length = args.window_length = 384 # for use with rythm features, otherwise window length = n_fft
    n_bins = args.n_bins = 84
    n_mels = args.n_mels = 128
    
    debug.log_info('Loading...')
    files = load_files(args.input)
    
    if process:
        batch = True
        if len(files) == 1:
            batch = False
        processor = Processor(args, mode='render', batch=batch)

    for file in files:
        args.input = file
        if dsp:
            try:
                if process:
                    args.y, args.sample_rate = sf.read(file, dtype='float32')
                    args.meta_data = utils.extract_metadata(file)
                    args.output = args.input
                else:
                    args.y=librosa.load(file, sr=args.sample_rate)[0]
            except RuntimeError:
                debug.log_error(f'Could not load {file}!')
                continue
            if not librosa.util.valid_audio(args.y):
                debug.log_error(f'{file} is not a valid audio file!')
                sys.exit()
            args.num_samples = args.y.shape[-1]
            args.duration = args.num_samples / args.sample_rate
            args.n_fft, args.hop_size, args.window_length, args.n_bins, args.n_mels = utils.adjust_anal_res(args)
            args.num_frames = int(args.num_samples / args.hop_size)
        
        if args.operation == "segment":
            debug.log_info(f'<Segmenting> {file}')
            craft = Segmentor(args)
            craft.main()
        elif args.operation == "extract":
            debug.log_info(f'<Extracting features> for {file}')
            craft = Extractor(args)
            craft.main()
        elif args.operation == "onset":
            debug.log_info(f'Detecting <onsets> for {file}')
            craft = OnsetDetector(args)
            craft.main()
        elif args.operation == "beat":
            debug.log_info(f'Detecting <beats> for {file}')
            craft = BeatDetector(args)
            craft.main()
        elif args.operation == "decomp":
            debug.log_info(f'<Decomposing> {file}')
            craft = Decomposer(args)
            craft.main()
        elif args.operation == "filter":
            debug.log_info(f'Applying <filter> to {file}')
            processor.filter(args.y, args.sample_rate, args.filter_frequency, args.filter_type)
        elif args.operation == "norm":
            debug.log_info(f'<Normalising> {file}')
            processor.normalise_audio(args.y, args.sample_rate, args.normalisation_level, args.normalisation_mode) 
        elif args.operation == "fade":
            debug.log_info(f'Applying> <fade to {file}')
            processor.fade_io(args.y, args.sample_rate, args.fade_in, args.fade_out, args.curve_type)  
        elif args.operation == "trim":
            debug.log_info(f'<Trimming> {file}')
            processor.trim_range(args.y, args.sample_rate, args.trim_range, args.fade_in, args.fade_out, args.curve_type)    
        elif args.operation == "pan":
            debug.log_info(f'<Panning> {file}')
            processor.pan(args.y, args.sample_rate, args.pan_value)
        elif args.operation == "mono":
            debug.log_info(f'Converting {file} to <mono>')
            processor.mono(args.y, args.sample_rate)             
        elif args.operation == "split":
            debug.log_info(f'<Splitting> {file}')
            processor.split(args.y, args.sample_rate, args.split_points)
            
        else:
            if args.operation == "wmeta":
                debug.log_info('Writing metadata')
                if(args.meta_file):
                    args.meta = utils.load_json(args.meta_file)
                else:
                    utils.error_debug.error('No metadata file provided!')
                    sys.exit()
                utils.write_metadata(file, args.meta)
            if args.operation == "rmeta":
                debug.log_info('Extracting metadata')
                utils.extract_metadata(file, args)
            
        args.n_fft = n_fft
        args.hop_size = hop_size
        args.window_length = window_length
        args.n_bins = n_bins
        args.n_mels = n_mels 

 
def load_files(input):
    files = []
    if input == None or input == '':
        debug.log_error('No input file or directory provided!')
        sys.exit()
    if input == '.':
        input = os.getcwd()
    # check if dir is home dir
    if input == os.path.expanduser('~'):
        debug.log_warning('You are about to process your home directory. Are you sure you want to continue?')
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
        debug.log_error('No valid files found!')
        sys.exit()
    return files




if __name__ == "__main__":
    
    utils.print_ascii_art()
    formatter_class=lambda prog: argparse.HelpFormatter(prog,
    max_help_position=8, width=80, indent_increment=4)
    usage = "wac.py operation [options] arg"
    parser = argparse.ArgumentParser(prog='Wave Craft', epilog="For more information, visit: https://github.com/tefdara/Wave-Craft",
                                     formatter_class=formatter_class, usage=usage)
    parser.add_argument("operation", type=str, choices=["segment", "extract", "proxim", "onset", "beat", "decomp", "filter", "norm", "fade", "trim", "split", "pan", "mono", "wmeta"], 
                    help="Operation to perform. See below for details on each operation.",
                    metavar='operation')
    parser.add_argument("input", type=str,
                    help="Path to the audio, metadata or dataset file. It can be a directory for batch processing. It is valid for all operations\n ")
    
    parser._action_groups[0].title = "Required arguments"
    parser._action_groups[1].title = "Help"
    
    # IO
    io_group = parser.add_argument_group('I/O')
    io_group.add_argument("-it", "--input-text", type=str, required=False,
                        help="The text file containing the segmentation data. Defaults to the nameofaudio.txt", metavar='')
    io_group.add_argument("-o", "--output-directory", type=str, default=None, help="Path to the output directory. Optional.", required=False, metavar='')
    io_group.add_argument("-st", "--save-txt", action='store_true', help="Save segment times to a text file.")
    
    # Audio settings
    audio_settings_group = parser.add_argument_group('Audio settings - these settings apply to all operations where relevant', description='')
    audio_settings_group.add_argument("-sr","--sample-rate", type=int, default=48000, help="Sample rate of the audio file. Default is 44100.", required=False, metavar='')
    audio_settings_group.add_argument("--fmin", type=float, default=30, help="Minimum analysis frequency. Default is 30.", required=False, metavar='')
    audio_settings_group.add_argument("--fmax", type=float, default=16000, help="Maximum analysis frequency. Default is 16000", required=False, metavar='')
    audio_settings_group.add_argument("--n-fft", type=int, default=2048, help="FFT size. Default is 2048.", required=False, metavar='')
    audio_settings_group.add_argument("--hop-size", type=int, default=512, help="Hop size. Default is 512.", required=False, metavar='')
    fo_def=30
    fi_def=20
    audio_settings_group.add_argument("-fi", "--fade-in", type=int, default=30, help=f"Duration in ms for fade in. Default is {fo_def}ms.", required=False, metavar='')
    audio_settings_group.add_argument("-fo", "--fade-out", type=int, default=50, help=f"Duration in ms for fade in. Default is {fi_def}ms.", required=False, metavar='')
    audio_settings_group.add_argument("-c", "--curve-type", type=str, choices=['exp', 'log', 'linear', 's_curve','hann'], default="exp",\
                        help="Type of curve to use for fades. Default is exponential.", required=False, metavar='')

    # Segmentation
    segmentation_group = parser.add_argument_group(title='Segmentation : splits the audio file into segments', description='operation -> segment')
    segmentation_group.add_argument("-m", "--segmentation-method", type=str, choices=["onset", "beat", "text"], required=False, default="onset",
                        help="Segmentation method to use.", metavar='')
    segmentation_group.add_argument("-ml", "--min-length", type=float, default=0.1, help="Minimum length of a segment in seconds. Default is 0.1s.\
                        anything shorter won't be used", required=False, metavar='')
    segmentation_group.add_argument("-t", "--onset-threshold", type=float, default=0.08, help="Onset detection threshold. Default is 0.08.", required=False, metavar='')
    segmentation_group.add_argument("-ts", "--trim-silence", type=float, default=-65, help="Trim silence from the beginning and end of the audio file. Default is -60 db.", required=False, metavar='')
    segmentation_group.add_argument("-oe", "--onset-envelope", type=str, choices=['mel', 'mfcc', 'cqt_chr', 'rms', 'zcr', 'cens', 'tmpg', 'ftmpg', 'tonnetz'], default="mel",\
                        help="Onset envelope to use for onset detection. Default is mel (mel spectrogram).\n Choices are: mel (mel spectrogram), mfcc (Mel-frequency cepstral coefficients), cqt_chr (chroma constant-Q transform), rms (root-mean-square energy), zcr (zero-crossing rate), cens (chroma energy normalized statistics), tmpg (tempogram), ftmpg (fourier tempogram), tonnetz (tonal centroid features)", required=False, metavar='')
    segmentation_group.add_argument("-bl", "--backtrack-length", type=float, default=20, help="Backtrack length in miliseconds. Backtracks the segments from the detected onsets. Default is 20ms.", required=False, metavar='')
                                    
    # Feature extraction
    feature_extraction_group = parser.add_argument_group(title='Feature extraction', description='operation -> extract')
    feature_extraction_group.add_argument("-fex", "--feature-extractor", type=str, choices=['mel', 'cqt', 'stft', 'cqt_chr', 'mfcc', 'rms', 'zcr', 'cens', 'tmpg', 'ftmpg', 'tonnetz', 'pf'], default=None,\
                                        help="Feature extractor to use. Default is all. Choices are: mel (mel spectrogram), cqt (constant-Q transform), stft (short-time Fourier transform), cqt_chr (chroma constant-Q transform), mfcc (Mel-frequency cepstral coefficients), rms (root-mean-square energy), zcr (zero-crossing rate), cens (chroma energy normalized statistics), tmpg (tempogram), ftmpg (fourier tempogram), tonnetz (tonal centroid features), pf (poly features).", required=False, metavar='')
    feature_extraction_group.add_argument("-fdic", "--flatten-dictionary", action='store_true', default=False, help="Flatten the dictionary of features. Default is False.", required=False)
    
    # Proximity metric calculation 
    proxi_metor_group = parser.add_argument_group(title='Distance metric learning - finds the most similar sounds based on a features dataset', description= 'operation -> proxim')
    proxi_metor_group.add_argument('-ns', '--n-similar', type=int, default=5,
                        help='Number of similar sounds to retrieve', metavar='')
    proxi_metor_group.add_argument('-id', '--identifier', type=str, required=False,
                        help='Identifier to test, i.e., the name of sound file. If not provided, all sounds in the dataset will be tested against each other', metavar='')
    proxi_metor_group.add_argument('-cls', '--class_to_analyse', type=str, default='stats',
                        help='Class to analyse. Default: stats. If not provided, all classes will be analysed. Note that this option can produce unexpected results if the dataset contains multiple classes with different dimensions', metavar='')
    proxi_metor_group.add_argument('-mt', '--metric-to-analyze', type=str, default=None,
                        help='Metric to analyze', metavar='')
    proxi_metor_group.add_argument('-ops', action='store_true', default=False,
                        help='Use opetions file to fine tune the metric learning')
    proxi_metor_group.add_argument('-mn', '--n-max', type=int, default=-1, 
                        help='Max number of similar files to retrieve, Default: -1 (all)', metavar='')
    proxi_metor_group.add_argument('-mtr', '--metric-range', type=float, default=None, nargs='+', 
                        help= 'Range of values to test for a specific metric. Default: None', metavar='')
                                   
    # Decomposition 
    decomposition_group = parser.add_argument_group(title='Decomposition - decomposes the audio file into harmonic, percussive or n components', description='operation -> decomp')
    decomposition_group.add_argument("-n", "--n-components", type=int, default=4, help="Number of components to use for decomposition.", required=False, metavar='')
    decomposition_group.add_argument("-hpss", "--source-separation", type=str, default=None, choices=["harmonic", "percussive"], help="Decompose the signal into harmonic and percussive components, before computing segments.", required=False, metavar='')
    
    # Beat detection 
    beat_group = parser.add_argument_group(title='Beat detection - detects beats in the audio file', description='operation -> beat')
    beat_group.add_argument("-k", type=int, default=5, help="Number of beat clusters to detect. Default is 5.", required=False, metavar='')
    
    # Filter 
    filter_group = parser.add_argument_group(title='Filter - applies a high / low pass filter to the audio file', description='operation -> filter')
    filter_group.add_argument("-ff", "--filter-frequency", type=int, default=40, 
                        help="Frequency to use for the high-pass filter. Default is 40 Hz. Set to 0 to disable", required=False, metavar='')
    filter_group.add_argument("-ft", "--filter-type", type=str, choices=['high', 'low'], default="high", 
                        help="Type of filter to use. Default is high-pass.", required=False, metavar='')

    # Normalization 
    normalization_group = parser.add_argument_group(title='Normalization - normalizes the audio file', description='operation -> norm')
    normalization_group.add_argument("-nl", "--normalisation-level", type=float, default=-3, required=False,
                        help="Normalisation level, default is -3 db.", metavar='')
    normalization_group.add_argument("-nm", "--normalisation-mode", type=str, default="peak", choices=["peak", "rms", "loudness"], 
                        help="Normalisation mode; default is 'peak'.", required=False, metavar='')
    
    # Metadata  
    metadata_group = parser.add_argument_group(title='Metadata - writes or reads metadata to/from the audio file', description='operations -> wmeta, rmeta')
    metadata_group.add_argument("--meta", type=str, help="List of metadata or comments to write to the file. Default is None.", required=False, nargs='+', metavar='')
    metadata_group.add_argument("-mf", "--meta-file", type=str, help="Path to a JSON metadata file. Default is None.", required=False, metavar='')

    # trim
    trim_group = parser.add_argument_group(title='Trim - trims the audio file', description='operation -> trim')
    trim_group.add_argument("-tr", "--trim-range", type=str, default=None, help="Trim position range in seconds. It can be a single value or a range (e.g. 0.5-1.5) or condition (e.g. -0.5).", required=False, metavar='')
    
    fade_group = parser.add_argument_group(title='Fade - applies a fade in and/or fade out to the audio file', description='operation -> fade')
    
    pan_group = parser.add_argument_group(title='Pan - pans the audio file', description='operation -> pan')
    pan_group.add_argument("-pv", "--pan-amount", type=float, default=0, help="Pan amount. Default is 0.", required=False, metavar='')
    
    split_group = parser.add_argument_group(title='Split - splits the audio file into multiple files', description='operation -> split')
    split_group.add_argument("-sp", "--split-points", type=float, default=None, help="Split points in seconds. It can be a single value or a list of split points (e.g. 0.5 0.2 3).", required=False, nargs='+', metavar='')
    
    args = parser.parse_args()
    
    revert = None        
    if not os.path.isdir(args.input) and not os.path.isfile(args.input):
        if(args.input == 'revert'):
            revert = True
        else:
            debug.log_error(f'Could not find input: {args.input}! Make sure the input is a valid  file or directory.')
            sys.exit()
    main(args, revert)
    
    