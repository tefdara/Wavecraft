"""
This module contains functions for extracting, generating, and writing metadata to audio files.
"""
import os
import subprocess
import sys
import tempfile
import json
import time
from .debug import Debug as debug


#######################
# Metadata
#######################

def extract_metadata(input_file):
    """
    Extracts the comment metadata from an audio file using ffprobe.

    Args:
        input_file (str): The path to the input audio file.

    Returns:
        str: The extracted comment metadata.

    Raises:
        None

    """
    # -show_entries format_tags=comment will show the comment metadata
    # -of default=noprint_wrappers=1:nokey=1 will remove the wrapper and the key from the output
    command = [
        'ffprobe',  input_file, '-v', 'error', '-show_entries', 'format_tags=comment', 
        '-of', 'default=noprint_wrappers=1:nokey=1',
    ]
    output = subprocess.check_output(command, stderr=subprocess.DEVNULL, universal_newlines=True)
    if 'not found' in output:
        debug.log_error('ffmpeg is not installed. Please install it if you want to copy the metadata over.')
        return None

    return output

def generate_metadata(input_file, args):
    """
    Generate metadata for the given input file.

    Args:
        input_file (str): The path to the input file.
        args (Namespace): The command line arguments.

    Returns:
        str: The final metadata string.
    """
    source_file_name = os.path.basename(input_file)
    creation_time = os.stat(input_file)
    # convert timestamp to a human readable format
    creation_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(creation_time.st_ctime))
    command = [
        'ffprobe',  input_file, '-v', 'error', '-show_entries', 
        'stream=sample_rate,channels,bits_per_raw_sample', '-of', 
        'default=noprint_wrappers=1:nokey=1'
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, universal_newlines=True)
    output, _ = process.communicate()

    base_data = {
        '-----------------------------------------': '',
        'wavecraft_version': '0.1.0',
        'operation': args.operation,
        'source_file_name': source_file_name,
        'source_creation_time': creation_time,
        'source_sample_rate': output.splitlines()[0],
        'source_channels': output.splitlines()[1],
        'source_bit_depth': output.splitlines()[2]
    }

    base_data = _stringify_dict(base_data)
    craft_data = _get_craft_metadata(args)
    craft_data = base_data + craft_data
    
    prev_metadata = extract_metadata(input_file)
    final_metadata = _concat_metadata(prev_metadata, craft_data)

    return final_metadata

def write_metadata(input_file, comment):
    """
    Writes metadata to an audio file.

    Args:
        input_file (str): The path to the input audio file.
        comment (str, list, dict): The metadata comment to be written. If it is a list, 
            the elements will be joined with newline characters.
            If it is a dictionary, the key-value pairs will be joined with newline characters.

    Returns:
        None
    """
        
    if input_file.endswith('.json'):
        debug.log_warning('Cannot write metadata to a JSON file, skipping...')
        return
    if isinstance(comment, list):
        comment = '\n'.join(comment)
    elif isinstance(comment, dict):
        comment = '\n'.join([f'{k} : {v}' for k, v in comment.items()])
        
    comment = comment.replace(',', '')
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        command = [
            'ffmpeg', '-v', 'quiet', '-y', '-i', input_file, '-metadata', f'comment={comment}', 
            '-codec', 'copy', tmp_file.name
        ]
        subprocess.run(command)
        os.replace(tmp_file.name, input_file)

def export_metadata(data, output_path, operation, suffix='metadata'):
    """
    Export metadata to a JSON file.

    Args:
        data (str): The metadata string to be exported.
        output_path (str): The path where the JSON file will be saved.
        suffix (str): The suffix to be appended to the JSON file name.

    Returns:
        None
    """
    output_path = os.path.realpath(output_path)
    meta_dir = os.path.dirname(output_path) + '/wavecraft_data'

    if not os.path.exists(meta_dir):
        os.makedirs(meta_dir)

    output_file = os.path.join(meta_dir, os.path.basename(output_path)
                               +f'_{operation}'+f'_{suffix}.json')


    data = data.replace('\n', ',')
    data_dict = {}
    data = data.split(',')
    for item in data:
        item = item.split(':')
        if len(item) >= 2:
            data_dict[item[0].strip()] = item[1].strip()
        else:
            data_dict[item[0].strip()] = ''
    # output_file = meta_dir+f'_{suffix}.json'
    if os.path.exists(output_file):
        debug.log_warning(f'Overwriting JSON metadata {os.path.basename(output_file)}...')
    else:
        debug.log_info(f'Exporting JSON metadata {os.path.basename(output_file)}...')
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data_dict, file, indent=4)


#######################
# Private functions
#######################

def _get_craft_metadata(args):

    normalization_metadata = _stringify_dict(_generate_normalization_metadata(args))
    filter_metadata = _stringify_dict(_generate_filter_metadata(args))
    trim_metadata = _stringify_dict(_generate_trim_metadata(args))
    fade_metadata = _stringify_dict(_generate_fade_metadata(args))
    audio_metadata = _stringify_dict(_generate_audio_settings_metadata(args))

    if args.operation == 'segment':
        seg_metadata = _stringify_dict(_generate_segmentation_metadata(args))
        metadata = [seg_metadata, normalization_metadata, filter_metadata, 
                    trim_metadata, fade_metadata, audio_metadata]
        metadata = _join_metadata(metadata)
        return metadata
    elif args.operation == 'extract':
        fex_metadata = _stringify_dict(_generate_feature_extraction_metadata(args))
        metadata = [fex_metadata, audio_metadata]
        metadata = _join_metadata(metadata)
        return metadata
    elif args.operation == 'decompose':
        decomp_metadata = _stringify_dict(_generate_decomposition_metadata(args))
        metadata = [decomp_metadata, audio_metadata]
        metadata = _join_metadata(metadata)
        return metadata
    elif args.operation == 'beat':
        beat_metadata = _stringify_dict(_generate_beat_detection_metadata(args))
        metadata = [beat_metadata, audio_metadata]
        metadata = _join_metadata(metadata)
        return metadata
    elif args.operation == 'filter':
        metadata = [filter_metadata, audio_metadata]
        metadata = _join_metadata(metadata)
        return metadata
    elif args.operation == 'norm':
        metadata = [normalization_metadata, audio_metadata]
        metadata = _join_metadata(metadata)
        return metadata
    elif args.operation == 'fade':
        metadata = [fade_metadata, audio_metadata]
        metadata = _join_metadata(metadata)
        return metadata
    elif args.operation == 'trim':
        metadata = [trim_metadata, audio_metadata]
        metadata = _join_metadata(metadata)
        return metadata
    elif args.operation == 'pan':
        pan_metadata = _stringify_dict(_generate_pan_metadata(args))
        metadata = [pan_metadata, audio_metadata]
        metadata = _join_metadata(metadata)
        return metadata
    elif args.operation == 'split':
        split_metadata = _stringify_dict(_generate_split_metadata(args))
        metadata = [split_metadata, audio_metadata]
        metadata = _join_metadata(metadata)
        return metadata
    elif args.operation == 'proxim':
        metadata = _stringify_dict(_generate_proximity_metric_metadata(args))
        return metadata

def _stringify_dict(d, new_line=True):
    if new_line:
        return '\n'.join([f'{k}:{v}' for k, v in d.items()])
    else:
        return ', '.join([f'{k}:{v}' for k, v in d.items()])

def _concat_metadata(meta_data, craft_data):

    if meta_data is None:
        meta_data = ''
        meta_data+=str(craft_data)
    else:
        for line in craft_data.splitlines():
            # if line in meta_data:
            #     # if the values are different then replace the old value with the new one??? 
            #     if line.split(':')[1].strip() != meta_data.split(':')[1].strip():
            #         # meta_data = meta_data.replace(line, '')
            #         # meta_data+=str(line)
            #         debug.log_warning(f'Metadata for line: {line} has changed. Keeping both values...')
            if line != craft_data.splitlines()[-1]:
                meta_data+=str(line)+'\n'
            else:
                meta_data+=str(line)
                
    return meta_data

def _join_metadata(meta_list):
    meta_data = ''
    for meta in meta_list:
        meta_data+=str(meta)+'\n'
    return meta_data

# Function for each category to receive args and return the correct dictionary
def _generate_io_metadata(args):
    return {
        'input_text': args.input_text,
        'output_directory': args.output_directory,
        'save_txt': args.save_txt
    }

def _generate_audio_settings_metadata(args):
    return {
        'sample_rate': args.sample_rate,
        'fmin': args.fmin,
        'fmax': args.fmax,
        'n_fft': args.n_fft,
        'hop_size': args.hop_size,
        'spectogram': args.spectogram,
        'no_resolution_adjustment': args.no_resolution_adjustment
    }

def _generate_segmentation_metadata(args):
    return {
        'seg_method': args.segmentation_method,
        'seg_min_length': args.min_length,
        'seg_onset_threshold': args.onset_threshold,
        'seg_onset_envelope': args.onset_envelope,
        'seg_backtrack_length': args.backtrack_length
    }

def _generate_feature_extraction_metadata(args):
    return {
        'fex_extractor': args.feature_extractor,
        'fex_flatten_dict': args.flatten_dictionary
    }

def _generate_proximity_metric_metadata(args):
    return {
        'prox_n_similar': args.n_similar,
        'prox_identifier': args.identifier,
        'prox_class_to_analyse': args.class_to_analyse,
        'prox_metric_to_analyze': args.metric_to_analyze,
        'prox_test_condition': args.test_condition,
        'prox_ops': args.ops,
        'prox_n_max': args.n_max,
        'prox_metric_range': args.metric_range
    }

def _generate_decomposition_metadata(args):
    return {
        'decomp_n_components': args.n_components,
        'decomp_source_separation': args.source_separation,
        'decomp_sklearn': args.sklearn,
        'decomp_nn_filter': args.nn_filter
    }

def _generate_beat_detection_metadata(args):
    return {
        'beat_k': args.k
    }

def _generate_filter_metadata(args):
    return {
        'filter_frequency': args.filter_frequency,
        'filter_type': args.filter_type
    }

def _generate_normalization_metadata(args):
    return {
        'norm_level': args.normalisation_level,
        'norm_mode': args.normalisation_mode
    }

def _generate_metadata_metadata(args):
    return {
        'meta_file': args.meta_file
    }

def _generate_trim_metadata(args):
    return {
        'trim_range': args.trim_range,
        'trim_silence': args.trim_silence
    }

def _generate_split_metadata(args):
    return {
        'split_points': args.split_points
    }

def _generate_fade_metadata(args):
    return {
        'fade_in': args.fade_in,
        'fade_out': args.fade_out,
        'curve_type': args.curve_type
    }

def _generate_pan_metadata(args):
    return {
        'pan_amount': args.pan_amount,
        'mono': args.mono
    }
        