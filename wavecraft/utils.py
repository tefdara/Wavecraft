import os, time
import numpy as np 
import subprocess, tempfile, json


# Define color codes for print messages
class bcolors:
    WHITE = '\033[97m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    GREY = '\033[37m'
    ENDC = '\033[0m'


def extract_metadata(input_file, args):
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
    
    source_m, seg_m = generate_metadata(input_file, args)
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

def generate_metadata(input_file, args):
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
    segmentation_metadata['seg_method'] = args.segmentation_method
    if(args.segmentation_method == 'onset'):
        segmentation_metadata['seg_onset_envelope'] = args.onset_envelope
    segmentation_metadata['seg_normalise_mode'] = args.normalisation_mode
    segmentation_metadata['seg_normalise_level'] = args.normalisation_level
    segmentation_metadata['seg_fade_duration'] = args.fade_duration
    segmentation_metadata['seg_fade_curve'] = args.curve_type
    segmentation_metadata['seg_filter_frequency'] = args.filter_frequency
    segmentation_metadata['seg_filter_type'] = args.filter_type + 'pass'
    segmentation_metadata['seg_onset_threshold'] = args.onset_threshold
    segmentation_metadata['seg_hop_size'] = args.hop_size
    segmentation_metadata['seg_n_fft'] = args.n_fft
    segmentation_metadata['seg_source_separation'] = args.source_separation

    # source_metadata.update(segmentation_metadata)
    return source_metadata, segmentation_metadata

def write_metadata(input_file, comment):
    if isinstance(comment, list):
        # convert to a string
        comment = '\n'.join(comment)
    elif isinstance(comment, dict):
        # convert to a string
        comment = '\n'.join([f'{k} : {v}' for k, v in comment.items()])
    comment = comment.replace(',', '')
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        command = [
            'ffmpeg', '-v', 'quiet', '-y', '-i', input_file, '-metadata', f'comment={comment}', '-codec', 'copy', tmp_file.name
        ]
        subprocess.run(command)
        # Rename the temporary file to the original file
        os.replace(tmp_file.name, input_file)
    
        
def load_json (input):
    if os.path.isfile(input):
        try:
            with open (input, 'r') as file:
                data = json.load(file)
                return data
        except Exception as e:
            print(f'{bcolors.RED}Error loading JSON file {input}. {str(e)}{bcolors.ENDC}')
            return None
    elif os.path.isdir(input):
        data = {}
        for file in os.listdir(input):
            if file.endswith('.json'):
                try:
                    with open (os.path.join(input, file), 'r') as f:
                        data[file] = json.load(f)
                except Exception as e:
                    print(f'{bcolors.RED}Error loading JSON file {file}. {str(e)}{bcolors.ENDC}')
                    return None
        return data
        
def export_json(data, output_path, data_type='metadata'):
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
    

def scientific_notation_to_float(array, precision=2):
        
    if isinstance(array, np.ndarray):
        if array.size == 0:
            return array    
        arr=[]
        for x in array:
            # use precision to format the number
            arr.append('{:.2f}'.format(x))
        return arr
    # if its not an array then check to see if its a number 
    elif isinstance(array, float):
        if array == 0:
            return 0
        return '{:.2f}'.format(array)
    else:
        print(f'{bcolors.RED}Invalid input. Expected a numpy array or a float, got {type(array)}{bcolors.ENDC}')
        return None
        
def check_format(file):
    return file.split('.')[-1] in ['wav', 'aif', 'aiff', 'flac', 'ogg', 'mp3', 'json']


def nearest_power_of_2(x):
    """Find the nearest power of 2 less than or equal to x."""
    return 2**np.floor(np.log2(x))

def adjust_anal_res(args):
    if args.duration > 5.0:
        return args.n_fft, args.hop_size, args.window_length, args.n_bins, args.n_mels
    scale_factor = min(args.duration, 5.0) / 5.0  # scale based on a 5-second reference
    length = args.y.shape[-1]
    n_fft = int(nearest_power_of_2(args.n_fft * scale_factor))
    n_fft = min(n_fft, length)

    hop_size = int(n_fft / 4)  
    # win_length = int(n_fft * 0.75)  # set to 75% of n_fft as a starting point
    win_length = int(384 * (scale_factor*0.5))
    n_bins = max(12, int(84 * scale_factor))  # at least 12 bins (for PCA later)
    n_mels = int(128 * scale_factor)
    return n_fft, hop_size, win_length, n_bins, n_mels

def flatten_dict(d):
    items = {}
    for k, v in d.items():
        if isinstance(v, list):
            for idx, item in enumerate(v):
                stat_type = k.split('_')[-1]
                key_without_stat = k.split('_' + stat_type)[0]
                if idx < 10:
                    idx = f'0{idx}'
                indexed_key = f"{key_without_stat}_{idx}_{stat_type}"
                items[indexed_key] = item
        else:
            items[k] = v
    # make sure stats are represented in order of index
    items = {k: v for k, v in sorted(items.items(), key=lambda item: item[0])}        
    return items

def deep_float_conversion(data):
    """
    Recursively convert string numbers to floats in a nested dict.
    Args:
        data: The data to convert.
    Returns:
        The converted float.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = deep_float_conversion(value)
    elif isinstance(data, list):
        for index, value in enumerate(data):
            data[index] = deep_float_conversion(value)
    elif isinstance(data, str):
        try:
            # Check if the string can be converted to a float.
            return float(data)
        except ValueError:
            return data
    return data

def load_dataset(data_path):
    """
    Load the data from the data_path.
    Args:
        data_path: The path to the data directory.
    Returns:
        A list of dictionaries containing the data.
    """
    data_dicts = []
    if os.path.isdir(data_path):
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith(".json"):
                    try:
                        with open(os.path.join(root, file)) as json_file:
                            data_content = json.load(json_file)
                            
                            # Convert any string numbers to floats in the nested dict.
                            data_dicts.append(deep_float_conversion(data_content))
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON from file: {file}")
                    except Exception as e:
                        print(f"Error reading from file {file}. Error: {e}")
    else:
       raise ValueError("The data path must be a directory.")
    return data_dicts
