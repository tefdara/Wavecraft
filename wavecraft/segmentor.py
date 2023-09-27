import os, sys
import librosa
import logging
import argparse
import soundfile as sf
import utils
import asyncio
from wavecraft import utils, BeatDetector, OnsetDetector, Processor
class Segmentor:
    def __init__(self, args):
        self.args = args
        self.processor = Processor(args)
        self.args.output_directory = self.args.output_directory or os.path.splitext(self.args.input)[0] + '_segments'
        self.base_segment_path = os.path.join(self.args.output_directory, os.path.basename(self.args.input).split('.')[0])
        
            
    def render_segments(self, segments):
        print(f'\n{utils.colors.GREEN}Rendering segments...{utils.colors.ENDC}\n')
        y_m, sr_m = sf.read(self.args.input, dtype='float32')
        segment_times = librosa.frames_to_time(segments, sr=self.args.sample_rate, hop_length=self.args.hop_size, n_fft=self.args.n_fft)
        segment_samps = librosa.time_to_samples(segment_times, sr=sr_m)
        # backtrack 
        segment_samps = segment_samps - int(self.args.backtrack_length * sr_m / 1000)
        
        prev_metadata = utils.extract_metadata(self.args.input, self.args)
        count = 0
        for i in range(len(segment_samps)):
            if i == len(segment_samps) - 1:
                # last segment
                start_sample = segment_samps[i]
                end_sample = len(y_m)
            else:
                start_sample = segment_samps[i]
                end_sample = segment_samps[i + 1]
            segment = y_m[start_sample:end_sample]
            segment = self.processor.fade_io(audio=segment, sr=self.args.sample_rate, fade_out=60, curve_type=self.args.curve_type)
            segment = self.processor.trim_after_last_silence(segment, sr_m, top_db=self.args.trim_silence, frame_length=self.args.n_fft, hop_length=self.args.hop_size)
            # skip segments that are too short
            segment_length = round(len(segment) / sr_m, 4)
            if segment_length < self.args.min_length:
                print(f'{utils.colors.YELLOW}Skipping segment {i+1} because it\'s too short{utils.colors.ENDC} : {segment_length}s')
                continue 
            count += 1
            segment = self.processor.fade_io(audio=segment, sr=self.args.sample_rate, curve_type=self.args.curve_type, fade_in=self.args.fade_in, fade_out=self.args.fade_out)
            segment = self.processor.filter(segment, self.args.sample_rate, self.args.filter_frequency, btype=self.args.filter_type)
            segment = self.processor.normalise_audio(segment, self.args.sample_rate, self.args.normalisation_level, self.args.normalisation_mode)
            segment_path = self.base_segment_path+f'_{count}.wav'
            
            sf.write(segment_path, segment, sr_m, format='WAV', subtype='PCM_24')
            print(f'{utils.colors.CYAN}Saving segment {count} to {segment_path}.{utils.colors.ENDC} {utils.colors.BLUE}length: {segment_length}s{utils.colors.ENDC}\n')
            utils.write_metadata(segment_path, prev_metadata)

        utils.export_metadata(prev_metadata, self.base_segment_path, data_type='seg_metadata')
                
        
        print(f'\n[{utils.colors.GREEN}Done{utils.colors.ENDC}]\n')


    def save_segments_as_txt(self, onsets):
        print(f'\n{utils.colors.GREEN}Saving segments as text file...{utils.colors.ENDC}')
        text_file_path = self.base_segment_path + '_segments.txt'
        with open (text_file_path, 'w') as file:
            for i in range(len(onsets) - 1):
                start_sample = onsets[i]
                end_sample = onsets[i + 1]
                # convert the sample indices to time in seconds
                # round the values to 6 decimal places but make sure there is at least 6 decimal places and add 0s if necessary
                start_time = round(librosa.samples_to_time(start_sample, sr=self.args.sample_rate), 6)
                start_time = f'{start_time:.6f}'
                end_time = round(librosa.samples_to_time(end_sample, sr=self.args.sample_rate), 6)
                end_time = f'{end_time:.6f}'
                # add a tab character between the start and end times
                file.write(f'{start_time}\t{end_time}\n')
        
        print(f'\n[{utils.colors.GREEN}Done{utils.colors.ENDC}]\n')
        
    def segment_using_txt(self, audio_path, txt_path, output_folder, file_format):
        
        y, sr = librosa.load(audio_path, sr=None)

        # Read the text file and split the audio based on the segments provided
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
                segment = utils.fade_io(segment, sr, fade_duration=self.args.fade_duration)
                segment_path = os.path.join(output_folder, f"segment_{i}.wav")
                sf.write(segment_path, segment, sr, format='WAV', subtype='FLOAT')
                
    def main(self):
        if self.args.segmentation_method == 'onset':
            detector = OnsetDetector(self.args)
            segments = asyncio.run(detector.main())
        elif self.args.segmentation_method == 'beat':
            detector = BeatDetector(self.args)
            segments = detector.main()
        
        user_input = input(f'{utils.colors.GREEN}Choose an action:{utils.colors.ENDC}\n1) Render segments\n2) Export segments as text file\n3) Exit\n')
        if user_input.lower() == '3':
            sys.exit()

        if self.args.segmentation_method == 'text':
            if(not self.args.input_text):
                self.args.input_text = os.path.splitext(self.args.input)[0] + '.txt'
            self.segment_using_txt(self.args.input, self.args.input_text, self.args.output_directory, self.args.file_format)
        else:
            os.makedirs(self.args.output_directory, exist_ok=True)
            if user_input.lower() == '1':
                self.render_segments(segments)
            elif user_input.lower() == '2':
                self.save_segments_as_txt(segments)


        if self.args.save_txt:
            self.save_segments_as_txt(segments, self.args.output_directory, self.args.sr)
            

    