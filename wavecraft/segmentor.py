import os
import sys
import asyncio
import librosa
import soundfile as sf
from wavecraft.debug import colors
from wavecraft.debug import Debug as debug
from . import utils

class Segmentor:
    """
    Class representing a segmentor for audio files.

    Args:
        args: An object containing the arguments for segmenting audio.

    Attributes:
        args: An object containing the arguments for segmenting audio.
        processor: An instance of the Processor class for audio processing.
        base_segment_path: The base path for saving the segmented audio files.

    Methods:
        render_segments: Renders the segments of the audio file.
        save_segments_as_txt: Saves the segments as a text file.
        segment_using_txt: Segments the audio file using a text file.
        main: The main method for segmenting audio based on the chosen segmentation method.
    """
    def __init__(self, args):
        self.args = args
        from .processor import Processor
        self.processor = Processor(args)
        self.args.output_directory = self.args.output_directory or os.path.splitext(self.args.input)[0] + '_segments'
        self.base_segment_path = os.path.join(self.args.output_directory, os.path.basename(self.args.input).split('.')[0])
            
    def render_segments(self, segments):
        """
        Renders the segments of the audio file.

        Args:
            segments: A list of segment indices.

        Returns:
            None
        """
        debug.log_info('Rendering segments...')
        
        y_m, sr_m = sf.read(self.args.input, dtype='float32')
        segment_times = librosa.frames_to_time(segments, sr=self.args.sample_rate, hop_length=self.args.hop_size, n_fft=self.args.n_fft)
        segment_samps = librosa.time_to_samples(segment_times, sr=sr_m)
        
        # backtrack
        segment_samps = segment_samps - int(self.args.backtrack_length * sr_m / 1000)
        
        # meta_data = utils.generate_metadata(self.args.input, self.args)
        
        count = 0
        for i, segment_samp in enumerate(segment_samps):
            
            if i == len(segment_samps) - 1:
                # last segment
                start_sample = segment_samp
                end_sample = len(y_m)
            else:
                start_sample = segment_samp
                end_sample = segment_samps[i + 1]
                
            segment = y_m[start_sample:end_sample]
            segment = self.processor.fade_io(segment, sr_m, fade_out=60, curve_type=self.args.curve_type)
            segment = self.processor.trim_silence_tail(segment, sr_m, top_db=self.args.trim_silence, frame_length=self.args.n_fft, hop_length=self.args.hop_size)
            # skip segments that are too short
            segment_length = round(len(segment) / sr_m, 4)
            if segment_length < self.args.min_length:
                debug.log_warning(f'Skipping segment {i+1} because it\'s too short: {segment_length}s')
                continue 
            count += 1
            segment = self.processor.fade_io(segment, sr_m, curve_type=self.args.curve_type, fade_in=self.args.fade_in, fade_out=self.args.fade_out)
            segment = self.processor.filter(segment, sr_m, self.args.filter_frequency, btype=self.args.filter_type)
            segment = self.processor.normalise_audio(segment, sr_m, self.args.normalisation_level, self.args.normalisation_mode)
            segment_path = self.base_segment_path+f'_{count}.wav'
            
            short_path = os.path.basename(segment_path)
            sf.write(segment_path, segment, sr_m, format='WAV', subtype='PCM_24')
            debug.log_info(f'Saving segment <{count}> to {short_path}. <length: {segment_length}s>')
            # utils.write_metadata(segment_path, meta_data)

        # utils.export_metadata(meta_data, self.base_segment_path, data_type='seg_metadata')
                
        
        debug.log_done(f'Exported {count} segments.')

    def save_segments_as_txt(self, onsets):
        """
        Saves the segments as a text file.

        Args:
            onsets: A list of segment onset indices.

        Returns:
            None
        """
        debug.log_info(f'Saving segments as text file to {self.args.output_directory}...')
        text_file_path = self.base_segment_path + '_segments.txt'
        with open (text_file_path, 'w') as file:
            for i in range(len(onsets) - 1):
                start_sample = onsets[i]
                end_sample = onsets[i + 1]
                start_time = round(librosa.samples_to_time(start_sample, sr=self.args.sample_rate), 6)
                start_time = f'{start_time:.6f}'
                end_time = round(librosa.samples_to_time(end_sample, sr=self.args.sample_rate), 6)
                end_time = f'{end_time:.6f}'
                file.write(f'{start_time}\t{end_time}\n')
        
    def segment_using_txt(self, audio_path, txt_path, output_folder):
        """
        Segments the audio file using a text file.

        Args:
            audio_path: The path to the audio file.
            txt_path: The path to the text file containing segment information.
            output_folder: The folder to save the segmented audio files.

        Returns:
            None
        """
        y, sr = librosa.load(audio_path, sr=None)

        with open(txt_path, 'r') as file:
            lines = file.readlines()

            for i, line in enumerate(lines):
                tokens = line.split()
                start_time, end_time = tokens[:2]
                start_time = float(start_time) * 1000  
                end_time = float(end_time) * 1000 

                start_sample = int(start_time * sr / 1000)
                end_sample = int(end_time * sr / 1000)
                segment = y[start_sample:end_sample]

                # Adding fade-in and fade-out effects
                segment = self.processor.fade_io(segment, sr, fade_in=self.args.fade_in, fade_out=self.args.fade_out, curve_type=self.args.curve_type)
                segment_path = os.path.join(output_folder, f"segment_{i}.wav")
                sf.write(segment_path, segment, sr, format='WAV', subtype='FLOAT')
                
    def main(self):
        """
        The main method for segmenting audio based on the chosen segmentation method.

        Returns:
            None
        """
        if self.args.segmentation_method == 'onset':
            from .onset_detector import OnsetDetector
            detector = OnsetDetector(self.args)
            segments = asyncio.run(detector.main())
        elif self.args.segmentation_method == 'beat':
            from .beat_detector import BeatDetector
            detector = BeatDetector(self.args)
            segments = detector.main()
        
        user_input = input(f'{colors.GREEN}Choose an action:{colors.ENDC}\n1) Render segments\n2) Export segments as text file\n3) Exit\n')
        if user_input.lower() == '3':
            sys.exit()

        if self.args.segmentation_method == 'text':
            if(not self.args.input_text):
                self.args.input_text = os.path.splitext(self.args.input)[0] + '.txt'
            self.segment_using_txt(self.args.input, self.args.input_text, self.args.output_directory)
        else:
            os.makedirs(self.args.output_directory, exist_ok=True)
            if user_input.lower() == '1':
                self.render_segments(segments)
            elif user_input.lower() == '2':
                self.save_segments_as_txt(segments)


        if self.args.save_txt:
            self.save_segments_as_txt(segments)
            

    