import librosa
import soundfile as sf
import numpy as np
import os
from scipy.signal import butter, filtfilt
from pyloudnorm import Meter, normalize
import wavecraft.utils as utils


def mode_handler(func):
    """Decorator to handle processing or rendering based on mode."""
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        if self.mode == "render":
            self._render(result)
        else:
            return result
    return wrapper

class Processor:
    def __init__(self, args, mode='raw'):
        self.args = args
        self.mode = mode
    
    def _render(self, y):
        sf.write(self.args.output, y, self.args.sample_rate, format='WAV', subtype='PCM_24')
    
    def render_components(self, components, activations, n_components, phase, render_path, sr=48000, hop_length=512):
        
        if not os.path.exists(render_path):
            os.makedirs(render_path)
        
        for i in range(n_components):
            # Reconstruct the spectrogram of the component
            component_spectrogram = components[:, i:i+1] @ activations[i:i+1, :]
            # Combine magnitude with the original phase
            component_complex = component_spectrogram * phase
            # Get the audio signal back using inverse STFT
            y_comp = librosa.istft(component_complex, hop_length=hop_length)
            
            # Save the component to an audio file
            sf.write(os.path.join(render_path, f'component_{i}.wav'), y_comp, sr)

    def render_hpss(self, y_harmonic, y_percussive, render_path, sr=48000):
        if not os.path.exists(render_path):
            os.makedirs(render_path)
            
        sf.write(os.path.join(render_path, 'harmonic.wav'), y_harmonic, sr)
        sf.write(os.path.join(render_path, 'percussive.wav'), y_percussive, sr)    

    @mode_handler
    def fade_io(self, audio, sr, fade_in=0, fade_out=0, curve_type='exp'):
        if fade_in == 0 and fade_out == 0:
            return audio
        
        if fade_in == 0:
            fade_in = 1
        if fade_out == 0:
            fade_out = 1
        
        max_len_percent = len(audio)*0.08 # 8% 
        # convert fade duration to samples
        fade_in_samples = int(fade_in * sr / 1000)
        fade_in_samples = int(min(fade_in_samples, max_len_percent))+1    
        fade_in_curve = utils.compute_curve(fade_in_samples, curve_type)
    
        fade_out_samples = int(fade_out * sr / 1000)
        fade_out_samples = int(min(fade_out_samples, max_len_percent))+1
        fade_out_curve = utils.compute_curve(fade_out_samples, curve_type)
       
        if len(audio.shape) == 1:
            audio[:fade_in_samples] *= fade_in_curve
            audio[-fade_out_samples:] *= fade_out_curve[::-1]
        else:
            for ch in range(audio.shape[1]):
                audio[:fade_in_samples, ch] *= fade_in_curve
                audio[-fade_out_samples:, ch] *= fade_out_curve[::-1] # reverse the curve for fade-out
            
        return audio

    
    @mode_handler
    def filter(self, data, sr, cutoff, btype='high', order=5):
        if(cutoff == 0):
            return data
        
        if len(data.shape) == 1:
            num_channels = 1
        else:
            num_channels = data.shape[1]

        # get the nyquist frequency
        nyq = 0.5 * sr
        # normalize the cutoff frequency
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype=btype, analog=False)

        # Initialize a new array for the filtered data
        y = np.zeros_like(data)

        # if stereo, filter each channel separately
        if num_channels > 1:
            for ch in range(num_channels):
                y[:, ch] = filtfilt(b, a, data[:, ch])
        else: # if mono
            y = filtfilt(b, a, data)

        return y

    @mode_handler
    def normalise_audio(self, y, sr, target_level, mode="peak"):
        """
        Normalize audio to a specified level based on the chosen normalization type.

        Parameters:
        - input_file: Path to the input audio file.
        - output_file: Path to save the normalized audio file.
        - normalization_type: Type of normalization ("peak", "rms", "loudness").
        - target_level: Desired level (in dB) for normalization.
        """

        if mode == "peak":
            # return target_level * y / max(abs(y))
            return normalize.peak(y, target_level)
        elif mode == "rms":
            rms_original = (y**2).mean()**0.5
            scaling_factor = 10**(target_level/20) / rms_original
            return y * scaling_factor
        elif mode == "loudness":
            meter = Meter(sr, block_size=0.05)
            loudness = meter.integrated_loudness(y)
            return normalize.loudness(y, loudness, target_level)
        else:
            raise ValueError(f"Unknown normalization type: {mode}")
    
    @mode_handler
    def trim_ends(self, y, threshold=20, frame_length=2048, hop_length=512):
        if len(y.shape) == 1:
            yt, indices = librosa.effects.trim(y, top_db=threshold, frame_length=frame_length, hop_length=hop_length)
            return yt, indices
        elif y.shape[1] == 2:
            # Trim the left channel
            _, index_left = librosa.effects.trim(y[:, 0], top_db=threshold, frame_length=frame_length, hop_length=hop_length)

            # Trim the right channel
            _, index_right = librosa.effects.trim(y[:, 1], top_db=threshold, frame_length=frame_length, hop_length=hop_length)

            # Use the maximum of the start indices and the minimum of the stop indices to keep the channels synchronized
            start_idx = max(index_left[0], index_right[0])
            end_idx = min(index_left[1], index_right[1])

            # Extract synchronized trimmed audio based on the indices
            yt_stereo = y[start_idx:end_idx, :]

            return yt_stereo, (start_idx, end_idx)

        else:
            print(f'{utils.bcolors.RED}Invalid number of channels. Expected 1 or 2, got {y.shape[1]}. Skipping trim...{utils.bcolors.ENDC}')
    @mode_handler
    def trim_range(self, y, sr, start, end):
        start_idx = int(start * sr)
        end_idx = int(end * sr)
        return y[start_idx:end_idx]
    
    @mode_handler
    def trim_after_last_silence(self, y, sr, top_db=-70.0, frame_length=2048, hop_length=512):
        """
        Trim audio after the last prolonged silence.

        Parameters:
        - y (np.ndarray): Audio time series.
        - sr (int): Sampling rate of the audio file.
        - top_db (float): The threshold (in dB) below which the signal is regarded as silent.
        - frame_length (int): The number of samples in each frame.
        - hop_length (int): The number of samples between successive frames.
        - silence_min_duration (float): The minimum duration of silence in seconds required for trimming.

        Returns:
        - y_trimmed (np.ndarray): The audio time series after trimming.
        """
        dur = len(y) / sr
        silence_min_duration = dur * 0.05
        if len(y.shape) > 1 and y.shape[1] == 2:
            y_mono = np.mean(y, axis=1)
        else:
            y_mono = y

        # Calculate amplitude envelope
        envelope = librosa.power_to_db(np.abs(librosa.stft(y_mono, n_fft=frame_length, hop_length=hop_length)), ref=np.max)
        # Detect where the envelope is below the threshold
        silent_frames = np.where(envelope.mean(axis=0) < top_db)[0]
        # If there are no silent frames, just return the original audio
        if len(silent_frames) == 0:
            return y

        # Group consecutive silent frames into silence periods
        breaks = np.where(np.diff(silent_frames) > 1)[0]
        silence_periods = np.split(silent_frames, breaks+1)

        # If the last silence is long enough, trim everything after it
        last_silence = silence_periods[-1]
        if len(last_silence) * hop_length / sr > silence_min_duration:
            trim_index = last_silence[0] * hop_length
            y_trimmed = y[:trim_index]
            return y_trimmed

        return y
    
    @mode_handler
    def random_crop(self, y, sr, duration):
        assert (y.ndim <= 2)

        target_len = sr * duration
        y_len = y.shape[-1]
        start = np.random.choice(range(np.maximum(1, y_len - target_len)), 1)[0]
        end = start + target_len
        if y.ndim == 1:
            y = y[start:end]
        else:
            y = y[:, start:end]
        return y
    
    @mode_handler
    def resample(self, y, sr, target_sr):
        return librosa.resample(y, sr, target_sr)
    
    @mode_handler
    def dither(self, y, sr, noise_amplitude=1e-5):
        return y + np.random.normal(0, noise_amplitude, y.shape)

    def batch_rename(self, input_dir, output_dir, prefix, extension):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for i, file in enumerate(os.listdir(input_dir)):
            os.rename(os.path.join(input_dir, file), os.path.join(output_dir, f'{prefix}_{i}.{extension}'))
    
    def batch_copy(self, input_dir, output_dir, extension):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for i, file in enumerate(os.listdir(input_dir)):
            os.rename(os.path.join(input_dir, file), os.path.join(output_dir, f'{i}.{extension}'))
    
    def batch_delete(self, input_dir):
        for file in os.listdir(input_dir):
            os.remove(os.path.join(input_dir, file))
    
    
        


