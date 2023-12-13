"""
This module contains the Processor class, responsible for applying different 
DSP operaions to the audio signal.
"""
import os
import sys
import soundfile as sf
import numpy as np
from scipy.signal import butter, filtfilt
from pyloudnorm import Meter, normalize
import librosa
import sounddevice as sd
from .debug import Debug as debug
from .debug import colors
from .utils import compute_curve
from .metadata import write_metadata, export_metadata


def mode_handler(func):
    """Decorator to handle processing or rendering based on mode."""

    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        # If in render mode, preview the sound first
        output = os.path.splitext(self.args.input)[0] + ".wav"
        if self.mode == "render":
            if self.batch:
                debug.log_warning("Batch processing. Skipping preview...")
                self._render(result, output)
                return
            else:
                prev_result = np.copy(result)
                debug.log_info("Playing preview")
                # debug.log_info("Press 's' to skip preview")
                sd.play(prev_result, samplerate=self.args.sample_rate)
                sd.wait()

                while True:
                    confirmation = input(
                        f"\n{colors.GREEN}Do you want to render the results?{colors.ENDC}\n\n1) Render\n2) Replay preview\n3) Exit\n"
                    )
                    if confirmation.lower() == "1":
                        debug.log_info("Rendering")
                        self._render(result, output)
                        debug.log_info("Done!")
                        break
                    elif confirmation.lower() == "2":
                        debug.log_info("Replaying")
                        sd.play(prev_result, samplerate=self.args.sample_rate)
                        sd.wait()
                    elif confirmation.lower() == "3":
                        debug.log_warning("Aborting render")
                        sys.exit(1)
                    else:
                        debug.log_error(
                            "\nInvalid input! Choose one of the options below."
                        )
                        continue
                return
        else:
            return result

    return wrapper


class Processor:
    def __init__(self, args=None, mode="raw", batch=True):
        self.args = args
        self.mode = mode
        self.batch = batch

        if args is None:
            self.args = type("", (), {})()
            self.args.input = " "
            self.args.sample_rate = 48000
            self.args.normalisation_level = -3
            self.args.filter_frequency = 40
            self.args.fade_in = 30
            self.args.fade_out = 30


    def _render(self, y, file):
        sf.write(file, y, self.args.sample_rate, format="WAV", subtype="PCM_24")
        write_metadata(file, self.args.meta_data)
        output = self.args.output_directory or os.path.splitext(self.args.input)[0]
        export_metadata(self.args.meta_data, output, self.args.operation)

    #############################################
    # Fade
    #############################################

    def _fade_io_internal(self, y, sr, fade_in=0, fade_out=0, curve_type="exp"):
        if fade_in == 0 and fade_out == 0:
            return y

        if fade_in == 0:
            fade_in = 1
        if fade_out == 0:
            fade_out = 1

        dur = len(y) / sr
        if dur < 1.0:
            max_len_percent = len(y) * 0.08  # 8%
        elif dur < 2.0 and dur >= 1.0:
            max_len_percent = len(y) * 0.15  # 15%
        elif dur >= 2.0:
            max_len_percent = len(y) * 0.5  # 50%

        if self.mode == "render":
            if fade_in + fade_out > dur * 1000:
                debug.log_error(
                    "Requested fade duration is longer than audio length! Exiting..."
                )
                return y
            max_len_percent = len(y)

        # convert fade duration to samples
        fade_in_samples = int(fade_in * sr / 1000)
        fade_in_samples = int(min(fade_in_samples, max_len_percent)) + 1
        fade_in_curve = compute_curve(fade_in_samples, curve_type)

        fade_out_samples = int(fade_out * sr / 1000)
        fade_out_samples = int(min(fade_out_samples, max_len_percent)) + 1
        fade_out_curve = compute_curve(fade_out_samples, curve_type)

        if len(y.shape) == 1:
            y[:fade_in_samples] *= fade_in_curve
            y[-fade_out_samples:] *= fade_out_curve[::-1]
        else:
            for ch in range(y.shape[1]):
                y[:fade_in_samples, ch] *= fade_in_curve
                y[-fade_out_samples:, ch] *= fade_out_curve[
                    ::-1
                ]  # reverse the curve for fade-out

        return y

    @mode_handler
    def fade_io(self, y, sr, fade_in=20, fade_out=20, curve_type="exp"):
        """
        Apply fade-in and fade-out effects to the input audio.

        Args:
            y (np.ndarray): The input audio signal.
            sr (int): The sample rate of the audio signal.
            fade_in (float, optional): The duration of the fade-in effect in seconds. Defaults to 20.
            fade_out (float, optional): The duration of the fade-out effect in seconds. Defaults to 20.
            curve_type (str, optional): The type of curve to use for the fade effect. Defaults to 'exp'.

        Returns:
            np.ndarray: The audio signal with fade-in and fade-out effects applied.
        """
        return self._fade_io_internal(y, sr, fade_in, fade_out, curve_type=curve_type)

    #############################################
    # Filter
    #############################################

    @mode_handler
    def filter(self, data, sr, cutoff, btype="high", order=4):
        """
        Apply a digital filter to the input data.

        Parameters:
        data (ndarray): Input data to be filtered.
        sr (int): Sample rate of the input data.
        cutoff (float): Cutoff frequency of the filter.
        btype (str, optional): Type of the filter. Defaults to 'high'.
        order (int, optional): Order of the filter. Defaults to 4.

        Returns:
        ndarray: Filtered data.
        """
        if cutoff == 0:
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
        else:  # if mono
            y = filtfilt(b, a, data)

        return y

    #############################################
    # Normalisation
    #############################################

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
            rms_original = (y**2).mean() ** 0.5
            scaling_factor = 10 ** (target_level / 20) / rms_original
            return y * scaling_factor
        elif mode == "loudness":
            meter = Meter(sr, block_size=0.05)
            loudness = meter.integrated_loudness(y)
            return normalize.loudness(y, loudness, target_level)
        else:
            raise ValueError(f"Unknown normalization type: {mode}")

    #############################################
    # Trim
    #############################################

    @mode_handler
    def trim(self):
        """
        Trims the audio based on the specified parameters.

        If `trim_range` is not None, it trims the audio within the specified range and applies
        fade-in and fade-out effects using the given `fade_in`, `fade_out`, and `curve_type` parameters.

        If `trim_range` is None, it trims the silence tails from the audio using the `trim_silence`,
        `n_fft`, and `hop_size` parameters.

        Returns:
            The trimmed audio.

        """
        if self.args.trim_range is not None:
            return self._trim_range_interanl(
                self.args.y,
                self.args.sample_rate,
                self.args.trim_range,
                self.args.fade_in,
                self.args.fade_out,
                self.args.curve_type,
            )
        else:
            return self._trim_silence_tails_interanl(
                self.args.y,
                self.args.sample_rate,
                self.args.trim_silence,
                self.args.n_fft,
                self.args.hop_size,
            )

    @mode_handler
    def trim_ends(self, y, threshold=20, frame_length=2048, hop_length=512):
        """
        Trim the silent ends of the audio signal.

        Parameters:
        - y (ndarray): Input audio signal.
        - threshold (float): Threshold in decibels (dB) below which the signal is considered silent. Default is 20 dB.
        - frame_length (int): Length of each frame in samples. Default is 2048 samples.
        - hop_length (int): Number of samples between successive frames. Default is 512 samples.

        Returns:
        - yt (ndarray): Trimmed audio signal.
        - indices (tuple): Start and end indices of the trimmed audio signal.
        """
        if len(y.shape) == 1:
            yt, indices = librosa.effects.trim(
                y, top_db=threshold, frame_length=frame_length, hop_length=hop_length
            )
            return yt, indices
        elif y.shape[1] == 2:
            # Trim the left channel
            _, index_left = librosa.effects.trim(
                y[:, 0],
                top_db=threshold,
                frame_length=frame_length,
                hop_length=hop_length,
            )

            # Trim the right channel
            _, index_right = librosa.effects.trim(
                y[:, 1],
                top_db=threshold,
                frame_length=frame_length,
                hop_length=hop_length,
            )

            # Use the maximum of the start indices and the minimum of the stop indices to keep the channels synchronized
            start_idx = max(index_left[0], index_right[0])
            end_idx = min(index_left[1], index_right[1])

            # Extract synchronized trimmed audio based on the indices
            yt_stereo = y[start_idx:end_idx, :]

            return yt_stereo, (start_idx, end_idx)

        else:
            debug.log_error(
                "Invalid number of channels. Expected 1 or 2, got {}".format(y.shape[1])
            )

    def _trim_range_interanl(
        self, y, sr, range, fade_in=25, fade_out=25, curve_type="exp"
    ):
        # e.g 0.1-0.5 means trim between 0.1 and 0.5 seconds
        # e.g 0.1- means trim after 0.1 seconds
        # e.g -0.5 means trim before 0.5 seconds
        range = range.split("-")
        start = int(float(range[0]) * sr) if range[0] != "" else 0
        end = int(float(range[1]) * sr) if range[1] != "" else len(y)
        if end < start or start < 0 or end > len(y):
            debug.log_error("Invalid range! Skipping trim")
            sys.exit(1)

        y_trimmed = np.concatenate((y[0:start], y[end:]))
        y_trimmed = self._fade_io_internal(
            y_trimmed, sr, fade_in, fade_out, curve_type=curve_type
        )
        return y_trimmed

    @mode_handler
    def trim_range(self, y, sr, range, fade_in=25, fade_out=25, curve_type="exp"):
        return self._trim_range_interanl(
            y, sr, range, fade_in, fade_out, curve_type=curve_type
        )

    def _trim_silence_tails_interanl(
        self, y, sr, top_db=-70.0, frame_length=2048, hop_length=512
    ):
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
        envelope = librosa.power_to_db(
            np.abs(librosa.stft(y_mono, n_fft=frame_length, hop_length=hop_length)),
            ref=np.max,
        )
        # Detect where the envelope is below the threshold
        silent_frames = np.where(envelope.mean(axis=0) < top_db)[0]
        # If there are no silent frames, just return the original audio
        if len(silent_frames) == 0:
            return y

        # Group consecutive silent frames into silence periods
        breaks = np.where(np.diff(silent_frames) > 1)[0]
        silence_periods = np.split(silent_frames, breaks + 1)

        # If the last silence is long enough, trim everything after it
        last_silence = silence_periods[-1]
        if len(last_silence) * hop_length / sr > silence_min_duration:
            trim_index = last_silence[0] * hop_length
            y_trimmed = y[:trim_index]
            return y_trimmed

        return y

    @mode_handler
    def trim_silence_tail(self, y, sr, top_db=-70.0, frame_length=2048, hop_length=512):
        """
        Trim the silence tail from the audio waveform.

        Args:
            y (ndarray): The input audio waveform.
            sr (int): The sample rate of the audio waveform.
            top_db (float, optional): The threshold (in decibels) below which the audio is considered silence. Defaults to -70.0.
            frame_length (int, optional): The length of each frame (in samples) for computing the energy. Defaults to 2048.
            hop_length (int, optional): The number of samples between successive frames. Defaults to 512.

        Returns:
            ndarray: The trimmed audio waveform.
        """
        return self._trim_silence_tails_interanl(
            y, sr, top_db, frame_length, hop_length
        )

    @mode_handler
    def random_crop(self, y, sr, duration):
        """
        Randomly crops the audio waveform to a specified duration.

        Args:
            y (ndarray): The audio waveform.
            sr (int): The sample rate of the audio waveform.
            duration (float): The desired duration of the cropped waveform in seconds.

        Returns:
            ndarray: The cropped audio waveform.
        """
        assert y.ndim <= 2

        target_len = sr * duration
        y_len = y.shape[-1]
        start = np.random.choice(range(np.maximum(1, y_len - target_len)), 1)[0]
        end = start + target_len
        if y.ndim == 1:
            y = y[start:end]
        else:
            y = y[:, start:end]
        return y

    #############################################
    # Misc
    #############################################
    @mode_handler
    def resample(self, y, sr, target_sr):
        """
        Resamples an audio signal to a target sample rate.

        Parameters:
            y (ndarray): The audio signal.
            sr (int): The current sample rate of the audio signal.
            target_sr (int): The desired sample rate to resample to.

        Returns:
            ndarray: The resampled audio signal.
        """
        return librosa.resample(y, sr, target_sr)

    @mode_handler
    def dither(self, y, sr, noise_amplitude=1e-5):
        """
        Applies dithering to the input audio signal.

        Parameters:
        - y: numpy.ndarray
            The input audio signal.
        - sr: int
            The sample rate of the audio signal.
        - noise_amplitude: float, optional
            The amplitude of the dithering noise. Default is 1e-5.

        Returns:
        - numpy.ndarray
            The dithered audio signal.
        """
        return y + np.random.normal(0, noise_amplitude, y.shape)

    def batch_rename(self, input_dir, output_dir, prefix, extension):
        """
        Batch rename files in the input directory and move them to the output directory.

        Args:
            input_dir (str): The path to the input directory.
            output_dir (str): The path to the output directory.
            prefix (str): The prefix to be added to the renamed files.
            extension (str): The file extension for the renamed files.

        Returns:
            None
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for i, file in enumerate(os.listdir(input_dir)):
            os.rename(
                os.path.join(input_dir, file),
                os.path.join(output_dir, f"{prefix}_{i}.{extension}"),
            )

    def batch_copy(self, input_dir, output_dir, extension):
        """
        Batch copies files from the input directory to the output directory with a specified extension.

        Args:
            input_dir (str): The path to the input directory.
            output_dir (str): The path to the output directory.
            extension (str): The file extension to be applied to the copied files.

        Returns:
            None
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for i, file in enumerate(os.listdir(input_dir)):
            os.rename(
                os.path.join(input_dir, file),
                os.path.join(output_dir, f"{i}.{extension}"),
            )

    def batch_delete(self, input_dir):
        """
        Deletes all files in the specified directory.

        Args:
            input_dir (str): The directory path where the files are located.
        """
        for file in os.listdir(input_dir):
            os.remove(os.path.join(input_dir, file))

    @mode_handler
    def pan(self, y, pan, mono):
        """
        Applies panning effect to the audio signal.

        Args:
            y (ndarray): Input audio signal.
            pan (float): Pan value ranging from -1 (full left) to 1 (full right).
            mono (bool): Flag indicating whether the input signal is mono.

        Returns:
            ndarray: Panned audio signal.
        """

        if mono:
            return self._mono_internal(y)

        if len(y.shape) == 1:
            y = np.expand_dims(y, axis=1)
        assert y.ndim == 2
        assert y.shape[1] == 2
        assert pan >= -1 and pan <= 1
        left = np.sqrt(0.5 * (1 - pan))
        right = np.sqrt(0.5 * (1 + pan))
        return np.hstack((left * y[:, 0:1], right * y[:, 1:2]))

    def _mono_internal(self, y):
        if len(y.shape) == 1:
            return y
        else:
            return np.mean(y, axis=1)

    def split(self, y, sr, split_points, name="split"):
        """
        Split the audio signal into multiple segments based on the given split points.

        Args:
            y (numpy.ndarray): The audio signal.
            sr (int): The sample rate of the audio signal.
            split_points (list): A list of split points, specified as fractions of the total duration.
            name (str, optional): The base name for the split files. Defaults to 'split'.

        Returns:
            None
        """
        split_points = [int(x * sr) for x in split_points]
        if len(split_points) == 1:
            debug.log_info("Rendering split files")
            y_1 = y[: split_points[0]]
            self._render(y_1, name + "_1.wav")
            y_2 = y[split_points[0] :]
            self._render(y_2, name + "_2.wav")
            return
        for i, split_point in enumerate(split_points):
            if i == 0:
                y_1 = y[: split_point]
                y_2 = y[split_point : split_points[i + 1]]
            else:
                y_1 = y[split_points[i - 1] : split_point]
                if i == len(split_points) - 1:
                    y_2 = y[split_point : len(y)]

            debug.log_info("Rendering split files")
            self._render(y_1, name + f"_{i}.wav")
            self._render(y_2, name + f"_{i+1}.wav")
