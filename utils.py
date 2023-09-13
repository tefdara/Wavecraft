import os, sys, time, librosa, soundfile as sf, numpy as np, subprocess, tempfile, json
from scipy.signal import butter, filtfilt
from pyloudnorm import Meter, normalize


# Define color codes for print messages
class bcolors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    CYAN = '\033[96m'
class Utils:

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
    import numpy as np

    def fade_io(self, audio, sr, fade_duration=20, curve_type='exp'):
        # convert fade duration to samples
        fade_duration_samples = int(fade_duration * sr / 1000)
        
        # If fade_duration_samples is larger than the segment, we should adjust it
        fade_duration_samples = min(fade_duration_samples, audio.shape[0])
        
        # switch between different curve types
        if curve_type == 'exp':
            fade_curve = np.linspace(0.0, 1.0, fade_duration_samples) ** 2
        elif curve_type == 'log':
            fade_curve = np.sqrt(np.linspace(0.0, 1.0, fade_duration_samples))
        elif curve_type == 'linear':
            fade_curve = np.linspace(0.0, 1.0, fade_duration_samples)
        elif curve_type == 's_curve':
            t = np.linspace(0.0, np.pi / 2, fade_duration_samples)
            fade_curve = np.sin(t)
        elif curve_type == 'hann':
            fade_curve = np.hanning(fade_duration_samples) / 2 + 0.5  # or fade_curve = 0.5 * (1 - np.cos(np.pi * np.linspace(0.0, 1.0, fade_duration_samples)))
        
        # Apply fade-in to the beginning and fade-out to the end for each channel
        for ch in range(audio.shape[1]): # For each channel
            audio[:fade_duration_samples, ch] *= fade_curve
            audio[-fade_duration_samples:, ch] *= fade_curve[::-1] # reverse the curve for fade-out
            
        return audio


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



    def extract_metadata(self, input_file, args):
        # this command will extract the comment metadata from the input file
        # -show_entries format_tags=comment will show the comment metadata
        # -of default=noprint_wrappers=1:nokey=1 will remove the wrapper and the key from the output
        command = [
            'ffprobe',  input_file, '-v', 'quiet', '-show_entries', 'format_tags=comment', '-of', 'default=noprint_wrappers=1:nokey=1',
        ]
        output = subprocess.check_output(command, stderr=subprocess.DEVNULL, universal_newlines=True)
        if 'not found' in output:
            print(f'{bcolors.RED}ffmpeg is not installed. Please install it if you want to copy the metadata over.{bcolors.ENDC}')
            return None
        
        source_m, seg_m = self.generate_metadata(input_file, args)
        # convert dicts to string and format it
        source_m = '\n'.join([f'{k} : {v}' for k, v in source_m.items()])
        seg_m = '\n'.join([f'{k} : {v}' for k, v in seg_m.items()])
        if output is None:
            output = ''
            output+=str(source_m)
            output+=str(seg_m)
        else:
            output+=str(seg_m)
            
        return output
    
    def generate_metadata(self, input_file, args):
        source_file_name = os.path.basename(input_file).split('.')[0]
        # get the file creation time and date from os
        creation_time = os.stat(input_file)
        # convert the timestamp to a human readable format
        creation_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(creation_time.st_ctime))
        # request the sample rate, bit depth and number of channels using ffprobe
        command = [
            'ffprobe',  input_file, '-v', 'quiet', '-show_entries', 'stream=sample_rate,channels,bits_per_raw_sample', '-of', 'default=noprint_wrappers=1:nokey=1'
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, universal_newlines=True)
        output, _ = process.communicate()
        # output = subprocess.check_output(command, stderr=subprocess.DEVNULL, universal_newlines=True)
        
        source_metadata = {}
        source_metadata['source_file_name'] = source_file_name
        source_metadata['source_creation_time'] = creation_time
        source_metadata['source_sample_rate'] = output.splitlines()[0]
        source_metadata['source_channels'] = output.splitlines()[1]
        source_metadata['source_bit_depth'] = output.splitlines()[2]
        segmentation_metadata = {}
        segmentation_metadata['seg_normalise_mode'] = args.normalisation_mode
        segmentation_metadata['seg_normalise_level'] = args.normalisation_level
        segmentation_metadata['seg_fade_duration'] = args.fade_duration
        segmentation_metadata['seg_fade_curve'] = args.curve_type
        segmentation_metadata['seg_filter_frequency'] = args.filter_frequency
        segmentation_metadata['seg_filter_type'] = args.filter_type + 'pass'
        segmentation_metadata['seg_onset_threshold'] = args.onset_threshold
        segmentation_metadata['seg_hop_size'] = args.hop_size
        segmentation_metadata['seg_n_fft'] = args.n_fft
        segmentation_metadata['seg_method'] = args.segmentation_method
        segmentation_metadata['seg_source_separation'] = args.source_separation

        # source_metadata.update(segmentation_metadata)
        return source_metadata, segmentation_metadata

    def write_metadata(self, input_file, comment):
        if isinstance(comment, list):
            # convert to a string
            comment = '\n'.join(comment)
        elif isinstance(comment, dict):
            # convert to a string
            comment = '\n'.join([f'{k} : {v}' for k, v in comment.items()])
        comment = comment.replace(',', '')
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            command = [
                'ffmpeg', '-y', '-i', input_file, '-metadata', f'comment={comment}', '-codec', 'copy', tmp_file.name
            ]
            subprocess.run(command)
            # Rename the temporary file to the original file
            os.replace(tmp_file.name, input_file)
        
            
    def load_json (self, input_file):
        # this command will extract the comment metadata from the input file
        with open (input_file, 'r') as file:
            data = json.load(file)
            return data
    def export_json(self, data, output_path, data_type='metadata'):
        data = data.replace('\n', ',')
        # convert to a dict
        data_dict = {}
        data_dict['id'] = os.path.basename(output_path)
        data = data.split(',')
        for item in data:
            item = item.split(':')
            data_dict[item[0].strip()] = item[1].strip()
        # data = dict(item.split(":") for item in data.split(","))
        output_file = output_path+f'_{data_type}.json'
        if os.path.exists(output_file):
            print(f'{bcolors.YELLOW}Overwriting JSON metadata {os.path.basename(output_file)}...{bcolors.ENDC}')
        else:
            print(f'{bcolors.CYAN}Exporting JSON metadata {os.path.basename(output_file)}...{bcolors.ENDC}')
        with open(output_file, 'w') as file:
            json.dump(data_dict, file, indent=4)
        
    def trim(self, y, threshold=20, frame_length=2048, hop_length=512):
        # Check if audio is stereo or mono
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
            print(f'{bcolors.RED}Invalid number of channels. Expected 1 or 2, got {y.shape[1]}. Skipping trim...{bcolors.ENDC}')
    
    def check_format(self, file):
        return file.split('.')[-1] in ['wav', 'aif', 'aiff', 'flac', 'ogg', 'mp3']