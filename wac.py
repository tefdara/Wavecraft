#!/usr/bin/env python3

import os, sys, argparse
from wavecraft import operator
from wavecraft import utils

if __name__ == "__main__":
    utils.print_ascii_art()
    formatter_class=lambda prog: argparse.HelpFormatter(prog,
    max_help_position=8, width=80, indent_increment=4)
    usage = "wac.py operation [options] arg"
    parser = argparse.ArgumentParser(prog='Wave Craft', epilog="For more information, visit: https://github.com/tefdara/Wave-Craft",
                                        formatter_class=formatter_class, usage=usage)
    parser.add_argument("operation", type=str, choices=["segment", "extract", "proxim", "onset", "beat", "decomp", "filter", "norm", "fade", "trim", "split", "pan", "wmeta"], 
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
    audio_settings_group = parser.add_argument_group('Audio Settings (low-level) - these apply to all operations where relevant', description='')
    audio_settings_group.add_argument("-sr","--sample-rate", type=int, default=22050, help="Sample rate that the files will be loaded in for processing.\
                                      Default is 22050. Note that the default for exported sounds is the sound files' native sample rate", required=False, metavar='')
    audio_settings_group.add_argument("--fmin", type=float, default=30, help="Minimum analysis frequency. Default is 30.", required=False, metavar='')
    audio_settings_group.add_argument("--fmax", type=float, default=11000, help="Maximum analysis frequency. Default is 11000", required=False, metavar='')
    audio_settings_group.add_argument("--n-fft", type=int, default=2048, help="FFT size. Default is 2048.", required=False, metavar='')
    audio_settings_group.add_argument("--hop-size", type=int, default=512, help="Hop size. Default is 512.", required=False, metavar='')
    audio_settings_group.add_argument("-spc", "--spectogram", type=str, choices=['mel', 'cqt', 'stft', 'cqt_chroma'], default=None,\
                        help="Spectogram to use when doing processes like decomposition among others. \
                            Default is None, in which case the appropiate spectogram will be used. Change this option only if you know what you are doing or if you want to experiment.", required=False, metavar='')
    audio_settings_group.add_argument("-nra", "--no-resolution-adjustment", action='store_true', default=False, 
                                      help="Disables the automatic adjusment of the analysis resolution and audio settings based on file duration. It is enabled by default.", required=False)
    
    # Segmentation
    segmentation_group = parser.add_argument_group(title='Segmentation : splits the audio file into segments', 
                                                   description='operation -> segment')
    segmentation_group.add_argument("-m", "--segmentation-method", type=str, choices=["onset", "beat", "text"], required=False, default="onset",
                        help="Segmentation method to use.", metavar='')
    segmentation_group.add_argument("-ml", "--min-length", type=float, default=0.1, help="Minimum length of a segment in seconds. Default is 0.1s.\
                        anything shorter won't be used", required=False, metavar='')
    segmentation_group.add_argument("-t", "--onset-threshold", type=float, default=0.08, help="Onset detection threshold. Default is 0.08.", required=False, metavar='')
    segmentation_group.add_argument("-oe", "--onset-envelope", type=str, choices=['mel', 'mfcc', 'cqt_chr', 'rms', 'zcr', 'cens', 'tmpg', 'ftmpg', 'tonnetz'], default="mel",\
                        help="Onset envelope to use for onset detection. Default is mel (mel spectrogram).\n Choices are: mel (mel spectrogram), mfcc (Mel-frequency cepstral coefficients), cqt_chr (chroma constant-Q transform), rms (root-mean-square energy), zcr (zero-crossing rate), cens (chroma energy normalized statistics), tmpg (tempogram), ftmpg (fourier tempogram), tonnetz (tonal centroid features)", required=False, metavar='')
    segmentation_group.add_argument("-bl", "--backtrack-length", type=float, default=40, help="Backtrack length in miliseconds. Backtracks the segments from the detected onsets. Default is 20ms.", required=False, metavar='')
    
    segmentation_generics = parser.add_argument_group(title='   generics', description='[-fi, --fade-in 30] [-fo, --fade-out 50] [-ct, --curve-type exp] [-ts, --trim-silence -65] [-ff, --filter-frequency 40] [-ft, --filter-type high] [-nl, --normalisation-level -3] [-nm, --normalisation-mode peak]')
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
    proxi_metor_group.add_argument('-tc', '--test-condition', type=str, default=None, help='Test condition for the specified metric. A condition is a string enclosed in \'\', that can be used to filter the dataset. For example, -mt duration -tc \'0.5-1.5\' or -mt duration -tc \'<0.5\'. \
                                   Default: None', metavar='')
    proxi_metor_group.add_argument('-ops', action='store_true', default=False,
                        help='Use opetions file to fine tune the metric learning')
    proxi_metor_group.add_argument('-mn', '--n-max', type=int, default=-1, 
                        help='Max number of similar files to retrieve, Default: -1 (all)', metavar='')
    proxi_metor_group.add_argument('-mtr', '--metric-range', type=float, default=None, nargs='+', 
                        help= 'Range of values to test for a specific metric. Default: None', metavar='')
                                    
    # Decomposition 
    decomposition_group = parser.add_argument_group(title='Decomposition - decomposes the audio file into harmonic, percussive or n components', description='operation -> decomp')
    decomposition_group.add_argument("-nc", "--n-components", type=int, default=4, help="Number of components to use for decomposition.", required=False, metavar='')
    decomposition_group.add_argument("-hpss", "--source-separation", type=str, default=None, choices=["harmonic", "percussive", 'hp'], help="Decompose the signal into harmonic and percussive components, If used for segmentation, the choice of both is invalid.", required=False, metavar='')
    decomposition_group.add_argument("-sk", "--sklearn", action='store_true', default=False, help="Use sklearn for decomposition. Default is False.", required=False)
    decomposition_group.add_argument("-nnf", "--nn-filter", action='store_true', default=False, 
                                     help="Use nearest neighbor filtering for decomposition. Default is False. Produces a single stream, n_components and hpss are not valid", required=False)
    
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
    trim_group = parser.add_argument_group(title='Trim - trims the audio file. Either range or silence can be used. Defining -tr will disable silence trimming', description='operation -> trim')
    trim_group.add_argument("-tr", "--trim-range", type=str, default=None, help="Trim position range in seconds. It can be a single value or a range (e.g. 0.5-1.5) or condition (e.g. -0.5).", required=False, metavar='')
    trim_group.add_argument("-ts", "--trim-silence", type=float, default=-65, help="Trim silence from the beginning and end of the audio file. Default is -65 db.", required=False, metavar='')

    
    # split
    split_group = parser.add_argument_group(title='Split - splits the audio file into multiple files', description='operation -> split')
    split_group.add_argument("-sp", "--split-points", type=float, default=None, help="Split points in seconds. It can be a single value or a list of split points (e.g. 0.5 0.2 3).", required=False, nargs='+', metavar='')
    
    # fade
    fade_group = parser.add_argument_group(title='Fade - applies a fade in and/or fade out to the audio file. See audio settings for options', description='operation -> fade')
    fo_def=30
    fi_def=20
    fade_group.add_argument("-fi", "--fade-in", type=int, default=30, help=f"Duration in ms for fade in. Default is {fo_def}ms.", required=False, metavar='')
    fade_group.add_argument("-fo", "--fade-out", type=int, default=50, help=f"Duration in ms for fade in. Default is {fi_def}ms.", required=False, metavar='')
    fade_group.add_argument("-ct", "--curve-type", type=str, choices=['exp', 'log', 'linear', 's_curve','hann'], default="exp",\
                        help="Type of curve to use for fades. Default is exponential.", required=False, metavar='')
    
    # pan
    pan_group = parser.add_argument_group(title='Pan - pans the audio file', description='operation -> pan')
    pan_group.add_argument("-pa", "--pan-amount", type=float, default=0, help="Pan amount. Default is 0.", required=False, metavar='')
    pan_group.add_argument("-mo", "--mono", action='store_true', default=False, help="Converts the audio file to mono.", required=False)

    args = parser.parse_args()

    revert = None        
    if not os.path.isdir(args.input) and not os.path.isfile(args.input):
        if(args.input == 'revert'):
            revert = True
        else:
            from wavecraft.debug import Debug as debug
            debug.log_error(f'Could not find input: {args.input}! Make sure the input is a valid  file or directory.')
            sys.exit()
    operator.main(args, revert)

