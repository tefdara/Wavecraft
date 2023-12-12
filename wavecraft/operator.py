import os, sys, asyncio
import librosa, soundfile as sf
from .segmentor import Segmentor
from .feature_extractor import Extractor
from .onset_detector import OnsetDetector
from .debug import Debug as debug
from .beat_detector import BeatDetector
from .decomposer import Decomposer
from .proxi_metor import ProxiMetor
from . import utils
from . import metadata


def main(args, revert=None):
    
    utils.progress(args.operation)
    if args.operation == "proxim":
        debug.log_info('Calculating <proximity metric>')
        craft = ProxiMetor(args)
        craft.main()
        return

    dsp = args.operation not in ["wmeta", "info"]
    process = args.operation not in ["segment", "extract", "onset", "beat", "decomp", "proxim"]
    # store these as they will be adjusted for short signals
    n_fft = args.n_fft
    hop_size = args.hop_size
    window_length = args.window_length = 384 # for use with rythm features, otherwise window length = n_fft
    n_bins = args.n_bins = 84
    n_mels = args.n_mels = 128
    
    debug.log_info('Loading...')
    files = load_files(args.input)
    
    warnings = {}
    errors = {}
    
    if process:
        batch = True
        if len(files) == 1:
            batch = False
        from .processor import Processor  
        processor = Processor(args, mode='render', batch=batch)

    for file in files:
        args.input = file
        if dsp:
            try:
                if process:
                    args.y, args.sample_rate = sf.read(file, dtype='float32')
                    args.meta_data = metadata.extract_metadata(file)
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
            if args.no_resolution_adjustment is False:
                debug.log_info('Adjusting analysis resolution for short signal...')
                args.n_fft, args.hop_size, args.window_length, args.n_bins, args.n_mels = utils.adjust_anal_res(args)
            args.num_frames = int(args.num_samples / args.hop_size)

        if args.operation == "segment":
            debug.log_info(f'<Segmenting> {file}')
            craft = Segmentor(args)
            craft.main()
        elif args.operation == "extract":
            debug.log_info(f'<Extracting features> for {file}')
            craft = Extractor(args)
            errors, warnings = craft.main()
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
            craft = Decomposer(args, True)
            asyncio.run(craft.main())
        elif args.operation == "filter":
            debug.log_info(f'Applying <filter> to {file}')
            processor.filter(args.y, args.sample_rate, args.filter_frequency,
                             btype=args.filter_type)
        elif args.operation == "norm":
            debug.log_info(f'<Normalising> {file}')
            processor.normalise_audio(args.y, args.sample_rate, args.normalisation_level,
                                      args.normalisation_mode)
        elif args.operation == "fade":
            debug.log_info(f'Applying> <fade to {file}')
            processor.fade_io(args.y, args.sample_rate, args.fade_in,
                              args.fade_out, args.curve_type)
        elif args.operation == "trim":
            debug.log_info(f'<Trimming> {file}')
            processor.trim()
        elif args.operation == "pan":
            debug.log_info(f'<Panning> {file}')
            processor.pan(args.y, args.pan_amount, args.mono)
        elif args.operation == "split":
            debug.log_info(f'<Splitting> {file}')
            processor.split(args.y, args.sample_rate, args.split_points)

        else:
            if args.operation == "wmeta":
                debug.log_info('Writing metadata')
                if args.meta_file:
                    args.meta = utils.load_json(args.meta_file)
                else:
                    debug.log_error('No metadata file provided!')
                    sys.exit()
                metadata.write_metadata(file, args.meta)
            if args.operation == "rmeta":
                debug.log_info('Extracting metadata')
                metadata.extract_metadata(file)

        meta = metadata.extract_metadata(file)
        print()
        args.n_fft = n_fft
        args.hop_size = hop_size
        args.window_length = window_length
        args.n_bins = n_bins
        args.n_mels = n_mels
        
    debug.log_done(f'<{args.operation}>')
    if len(warnings) > 0:
        debug.log_warning(f'Finished with <{len(warnings)} warning(s)>:')
        for k in warnings.keys():
            for w in warnings[k]:
                debug.log_warning(f'{k}: {w.message}. <Line {w.lineno} in file:> {w.filename}')
            
    if len(errors) > 0:
        debug.log_error(f'Finished with <{len(errors)} error(s)>:')
        for k in errors.keys():
            for e in errors[k]:
                debug.log_error(f'{k}: <{e}>')
 
def load_files(input_file):
    files = []
    if input_file is None or input_file == '':
        debug.log_error('No input file or directory provided!')
        sys.exit()
    if input_file == '.':
        input_file = os.getcwd()
    # check if dir is home dir
    if input_file == os.path.expanduser('~'):
        debug.log_warning('You are about to process your home directory. Are you sure you want to continue?')
        user_input = input_file('\n1) Yes\n2) No\n')
        if user_input.lower() == '2':
            sys.exit(1)           
    if os.path.isdir(input_file):
        input_dir = input_file
        for file in os.listdir(input_file):
            if utils.check_format(file):
                files.append(os.path.join(input_dir, file))
    # single file
    else:
        if utils.check_format(input_file):
            files.append(input_file)
    if len(files) == 0:
        debug.log_error('No valid files found!')
        sys.exit()
    return files


