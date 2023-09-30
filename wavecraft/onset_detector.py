import librosa
import os
import sys
import numpy as np
from wavecraft.debug import Debug as debug


class OnsetDetector:
    def __init__(self, args):
        self.args = args
        self.args.output_directory = self.args.output_directory or os.path.splitext(self.args.input)[0] + '_segments'
    
    def compute_onsets(self, y, sr, hop_length=512, n_fft=2048, fmin=27.5, fmax=16000., lag=2, max_size=3, env_method='mel'):
        
        if env_method == 'mel': 
            debug.log_info('Onset envelope: <mel spectrogram>')
            mel_spectogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=138, fmin=fmin, fmax=fmax)
            S = librosa.power_to_db(mel_spectogram, ref=np.max)
            o_env = librosa.onset.onset_strength(S=S,
                                                  sr=sr,
                                                  hop_length=hop_length,
                                                  lag=lag, max_size=max_size)
        elif env_method == 'rms':
            debug.log_info('Onset envelope: <rms>')
            S = np.abs(librosa.stft(y=y))
            rms = librosa.feature.rms(S=S, frame_length=n_fft, hop_length=hop_length)
            o_env = librosa.onset.onset_strength(S=rms[0],
                                                  sr=sr,
                                                  hop_length=hop_length,
                                                  lag=lag, max_size=max_size)
        elif env_method == 'cens':
            debug.log_info('Onset envelope: <chroma cens>')
            cens = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length, n_chroma=12, bins_per_octave=36)
            o_env = librosa.onset.onset_strength(S=cens,
                                                  sr=sr,
                                                  hop_length=hop_length,
                                                  lag=lag, max_size=max_size)
        elif env_method == 'cqt_chr':
            debug.log_info('Onset envelope: <chroma cqt>')
            cqt_chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length, n_chroma=12, bins_per_octave=36)
            o_env = librosa.onset.onset_strength(S=cqt_chroma,
                                                  sr=sr,
                                                  hop_length=hop_length,
                                                  lag=lag, max_size=max_size)
        elif env_method == 'mfcc':
            debug.log_info('Onset envelope: <mfcc>')
            mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13, n_mels=138, fmin=fmin, fmax=fmax)
            o_env = librosa.onset.onset_strength(S=mfcc,
                                                  sr=sr,
                                                  hop_length=hop_length,
                                                  lag=lag, max_size=max_size)
        elif env_method == 'tmpg':
            debug.log_info('Onset envelope: <tempogram>')
            tempogram = librosa.feature.tempogram(y=y, sr=sr, hop_length=hop_length, win_length=384, center=True)
            o_env = librosa.onset.onset_strength(S=tempogram,
                                                  sr=sr,
                                                  hop_length=hop_length,
                                                  lag=lag, max_size=max_size)
        elif env_method == 'ftmpg':
            debug.log_info('Onset envelope: <fourier tempogram>')
            fourier_tempogram = librosa.feature.fourier_tempogram(y=y, sr=sr, hop_length=hop_length, win_length=384, center=True)
            o_env = librosa.onset.onset_strength(S=fourier_tempogram,
                                                  sr=sr,
                                                  hop_length=hop_length,
                                                  lag=lag, max_size=max_size)
        elif env_method == 'tonnetz':
            debug.log_info('Onset envelope: <tonnetz>')
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr, hop_length=hop_length)
            o_env = librosa.onset.onset_strength(S=tonnetz,
                                                  sr=sr,
                                                  hop_length=hop_length,
                                                  lag=lag, max_size=max_size)
        elif env_method == 'zcr':
            debug.log_info('Onset envelope: <zero crossing rate>')
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
            debug.log_info('Decomposing the signal into harmonic and percussive components...')
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
            debug.log_info('No onsets detected! Try to reduce the onset threshold with -t')
            sys.exit(1)
            return
        ot = librosa.frames_to_time(onsets, sr=self.args.sample_rate, hop_length=self.args.hop_size)
        onset_times=[]
        for i in range(len(ot)):
            onset_times.append(round(ot[i], 5))
        
        segs = onset_times + [self.args.duration]
        segment_lengths = np.around(np.diff(segs),5).tolist()
        dur = round(self.args.duration, 5)
        
        print()
        debug.log_stat(f'Detected {len(onsets)} onsets:\n')
        debug.log_value(f'File length (sec): {dur}')
        debug.log_value(f'File length (min): {round(dur/60, 2)}')
        debug.log_value(f'Frame count: {self.args.num_frames}')

        if len(onsets) > 15:
            onsets_c = [onsets[0], onsets[1], onsets[2], onsets[-3], onsets[-2], onsets[-1]]
            onset_times_c = [onset_times[0], onset_times[1], onset_times[2], onset_times[-3], onset_times[-2], onset_times[-1]]
            segment_lengths_c = [segment_lengths[0], segment_lengths[1], segment_lengths[2], segment_lengths[-3], segment_lengths[-2], segment_lengths[-1]]
            onsets_c = [round(x, 3) for x in onsets_c]
            onset_times_c = [round(x, 3) for x in onset_times_c]
            segment_lengths_c = [round(x, 3) for x in segment_lengths_c]
            onsets_c.insert(3, '...')
            onset_times_c.insert(3, '...')
            segment_lengths_c.insert(3, '...')

        debug.log_value(f'Onset Frames: {onsets_c if len(onsets) > 12 else onsets}')
        debug.log_value(f'Onset Times: {onset_times_c if len(onsets) > 12 else onset_times}')
        debug.log_value(f'Segment Lengths: {segment_lengths_c if len(onsets) > 12 else segment_lengths}\n')
        
        return onsets
