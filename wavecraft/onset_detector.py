import argparse
import librosa
import os
import sys
import soundfile as sf
import numpy as np
import traceback
import asyncio
import wavecraft.utils as utils


class OnsetDetector:
    def __init__(self, args):
        self.args = args
        self.args.output_directory = self.args.output_directory or os.path.splitext(self.args.input)[0] + '_segments'
        self.info_logger = utils.get_logger('info', 'onset_info')
        self.error_logger = utils.get_logger('error', 'onset_error')
        self.warning_logger = utils.get_logger('warning', 'onset_warning')
        self.statlogger = utils.get_logger('stat', 'onset_stat')
        self.value_logger = utils.get_logger('value', 'onset_value')
    
    def compute_onsets(self, y, sr, hop_length=512, n_fft=2048, fmin=27.5, fmax=16000., lag=2, max_size=3, env_method='mel'):
        
        extra = utils.extra_log_string(prepend='Using', append='for onset detection.')
        
        if env_method == 'mel': 
            self.info_logger.info('mel spectrogram', extra=extra)
            mel_spectogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=138, fmin=fmin, fmax=fmax)
            S = librosa.power_to_db(mel_spectogram, ref=np.max)
            o_env = librosa.onset.onset_strength(S=S,
                                                  sr=sr,
                                                  hop_length=hop_length,
                                                  lag=lag, max_size=max_size)
        elif env_method == 'rms':
            self.info_logger.info('rms', extra=extra)
            S = np.abs(librosa.stft(y=y))
            rms = librosa.feature.rms(S=S, frame_length=n_fft, hop_length=hop_length)
            o_env = librosa.onset.onset_strength(S=rms[0],
                                                  sr=sr,
                                                  hop_length=hop_length,
                                                  lag=lag, max_size=max_size)
        elif env_method == 'cens':
            self.info_logger.info('chroma cens', extra=extra)
            cens = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length, n_chroma=12, bins_per_octave=36)
            o_env = librosa.onset.onset_strength(S=cens,
                                                  sr=sr,
                                                  hop_length=hop_length,
                                                  lag=lag, max_size=max_size)
        elif env_method == 'cqt_chr':
            self.info_logger.info('chroma cqt', extra=extra)
            cqt_chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length, n_chroma=12, bins_per_octave=36)
            o_env = librosa.onset.onset_strength(S=cqt_chroma,
                                                  sr=sr,
                                                  hop_length=hop_length,
                                                  lag=lag, max_size=max_size)
        elif env_method == 'mfcc':
            self.info_logger.info('mfcc', extra=extra)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13, n_mels=138, fmin=fmin, fmax=fmax)
            o_env = librosa.onset.onset_strength(S=mfcc,
                                                  sr=sr,
                                                  hop_length=hop_length,
                                                  lag=lag, max_size=max_size)
        elif env_method == 'tmpg':
            self.info_logger.info('tempogram', extra=extra)
            tempogram = librosa.feature.tempogram(y=y, sr=sr, hop_length=hop_length, win_length=384, center=True)
            o_env = librosa.onset.onset_strength(S=tempogram,
                                                  sr=sr,
                                                  hop_length=hop_length,
                                                  lag=lag, max_size=max_size)
        elif env_method == 'ftmpg':
            self.info_logger.info('fourier tempogram', extra=extra)
            fourier_tempogram = librosa.feature.fourier_tempogram(y=y, sr=sr, hop_length=hop_length, win_length=384, center=True)
            o_env = librosa.onset.onset_strength(S=fourier_tempogram,
                                                  sr=sr,
                                                  hop_length=hop_length,
                                                  lag=lag, max_size=max_size)
        elif env_method == 'tonnetz':
            self.info_logger.info('tonnetz', extra=extra)
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr, hop_length=hop_length)
            o_env = librosa.onset.onset_strength(S=tonnetz,
                                                  sr=sr,
                                                  hop_length=hop_length,
                                                  lag=lag, max_size=max_size)
        elif env_method == 'zcr':
            print(utils.bcolors.YELLOW + 'Using zero-crossing rate for onset detection.' + utils.bcolors.ENDC)
            zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=n_fft, hop_length=hop_length)
            o_env = librosa.onset.onset_strength(S=zcr,
                                                  sr=sr,
                                                  hop_length=hop_length,
                                                  lag=lag, max_size=max_size)
            
        onsets = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, hop_length=hop_length, backtrack=True, delta=self.args.onset_threshold)
        return onsets

    async def main(self):
        if self.args.hop_size == 512:
            self.args.hop_size = int(librosa.time_to_samples(1./200, sr=self.args.sample_rate))
        if self.args.source_separation is not None:
            # wait for the decomposition to finish
            from decomposer import Decomposer
            print(f'{utils.bcolors.YELLOW}Decomposing the signal into harmonic and percussive components...{utils.bcolors.ENDC}')
            decomposer = Decomposer(self.input, 'hpss', render=True, render_path=os.path.join(os.path.dirname(self.input), 'components'))
            H, P = await decomposer._decompose_hpss(self.args.y, n_fft=self.args.n_fft, hop_length=self.args.hop_size)
            if self.decompose == 'harmonic':
                self.args.y = H
            elif self.decompose == 'percussive':
                self.args.y = P
            else:
                raise ValueError('Invalid decomposition type.')
        
        onsets = self.compute_onsets(self.args.y, sr=self.args.sample_rate, n_fft=self.args.n_fft, hop_length=self.args.hop_size, fmin=self.args.fmin, fmax=self.args.fmax, env_method=self.args.onset_envelope)
        if len(onsets) == 0 or onsets is None:
            print(f'{utils.bcolors.RED}No onsets detected! Try to reduce the onset threshold with -t.{utils.bcolors.ENDC}')
            sys.exit(1)
            return
        ot = librosa.frames_to_time(onsets, sr=self.args.sample_rate, hop_length=self.args.hop_size)
        onset_times=[]
        for i in range(len(ot)):
            onset_times.append(round(ot[i], 5))
        
        segs = onset_times + [self.args.duration]
        segment_lengths = np.around(np.diff(segs),5).tolist()
        
        self.statlogger.info(f'Detected {len(onsets)} onsets:')
        length_val = utils.extra_log_value(round(self.args.duration, 5), 'seconds')
        self.value_logger.info(f'File length: ', extra=length_val)
        frame_val = utils.extra_log_value(self.args.num_frames,'frames')
        self.value_logger.info(f'Frame count: ', extra=frame_val)

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
        frames_val = utils.extra_log_value(onsets_c if len(onsets) > 12 else onsets, 'frames')
        times_val = utils.extra_log_value(onset_times_c if len(onsets) > 12 else onset_times, 'seconds')
        lengths_val = utils.extra_log_value(segment_lengths_c if len(onsets) > 12 else segment_lengths, 'seconds')
        self.value_logger.info(f'Onset Frames: ', extra=frames_val)
        self.value_logger.info(f'Onset times: ', extra=times_val)
        self.value_logger.info(f'Segment lengths: ', extra=lengths_val)
        print('')
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
        print(f'{utils.bcolors.YELLOW}Verbose mode enabled.{utils.bcolors.ENDC}')

    onset_detector = OnsetDetector(args)

    try:
        asyncio.run(onset_detector.main())
    except Exception as e:
        print(f'{utils.bcolors.RED}Error: {type(e).__name__}! {e}{utils.bcolors.ENDC}')
        traceback.print_exc()
        sys.exit(1)