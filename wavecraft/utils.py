import os, time
import numpy as np 
import subprocess, tempfile, json
import threading
from wavecraft.debug import Debug as debug

def concat_metadata(meta_data, craft_data):
        # print (meta_data)
    if meta_data is None:
        meta_data = ''
        meta_data+=str(craft_data)
    else:
        # check if the values are the same
        for line in craft_data.splitlines():
            if line in meta_data:
                # check if the values are the same
                if line.split(':')[1].strip() == meta_data.split(':')[1].strip():
                    continue
                else:
                    # if the values are different then replace the old value with the new one
                    meta_data = meta_data.replace(line, '')
                    debug.log_warning(f'Overwriting metadata {line}...')
                    meta_data+=str(line)
            else:
                if line != craft_data.splitlines()[-1]:
                    meta_data+=str(line)+'\n'
                else:
                    meta_data+=str(line)
    return meta_data

def extract_metadata(input_file):
    # this command will extract the comment metadata from the input file
    # -show_entries format_tags=comment will show the comment metadata
    # -of default=noprint_wrappers=1:nokey=1 will remove the wrapper and the key from the output
    command = [
        'ffprobe',  input_file, '-v', 'error', '-show_entries', 'format_tags=comment', '-of', 'default=noprint_wrappers=1:nokey=1',
    ]
    output = subprocess.check_output(command, stderr=subprocess.DEVNULL, universal_newlines=True)
    if 'not found' in output:
        debug.log_error('ffmpeg is not installed. Please install it if you want to copy the metadata over.')
        return None
    
    return output
    
def generate_metadata(input_file, args):
    source_file_name = os.path.basename(input_file)
    creation_time = os.stat(input_file)
    # convert the timestamp to a human readable format
    creation_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(creation_time.st_ctime))
    command = [
        'ffprobe',  input_file, '-v', 'error', '-show_entries', 'stream=sample_rate,channels,bits_per_raw_sample', '-of', 'default=noprint_wrappers=1:nokey=1'
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, universal_newlines=True)
    output, _ = process.communicate()
    
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
    segmentation_metadata['seg_fade_in_duration'] = args.fade_in
    segmentation_metadata['seg_fade_out_duration'] = args.fade_out
    segmentation_metadata['seg_fade_curve'] = args.curve_type
    segmentation_metadata['seg_filter_frequency'] = args.filter_frequency
    segmentation_metadata['seg_filter_type'] = args.filter_type + 'pass'
    segmentation_metadata['seg_onset_threshold'] = args.onset_threshold
    segmentation_metadata['seg_hop_size'] = args.hop_size
    segmentation_metadata['seg_n_fft'] = args.n_fft
    segmentation_metadata['seg_source_separation'] = args.source_separation

    # convert to string and
    source_metadata = '\n'.join([f'{k}:{v}' for k, v in source_metadata.items()])
    segmentation_metadata = '\n'.join([f'{k}:{v}' for k, v in segmentation_metadata.items()])
    craft_data = source_metadata + segmentation_metadata
    
    prev_metadata = extract_metadata(input_file)
    final_metadata = concat_metadata(prev_metadata, craft_data)
    
    return final_metadata

def write_metadata(input_file, comment):
    if input_file.endswith('.json'):
        return
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

def export_metadata(data, output_path, data_type='metadata'):
    data = data.replace('\n', ',')
    data_dict = {}
    data = data.split(',')
    for item in data:
        item = item.split(':')
        if len(item) >= 2:
            data_dict[item[0].strip()] = item[1].strip()
        else:
            data_dict[item[0].strip()] = ''
    output_file = output_path+f'_{data_type}.json'
    if os.path.exists(output_file):
        debug.log_warning(f'Overwriting JSON metadata {os.path.basename(output_file)}...')
    else:
        debug.log_info(f'Exporting JSON metadata {os.path.basename(output_file)}...')
    with open(output_file, 'w') as file:
        json.dump(data_dict, file, indent=4)
        
def load_json (input):
    if os.path.isfile(input):
        try:
            with open (input, 'r') as file:
                data = json.load(file)
                return data
        except Exception as e:
            debug.log_error(f'Error loading JSON file {input}. {str(e)}')
            return None
    elif os.path.isdir(input):
        data = {}
        for file in os.listdir(input):
            if file.endswith('.json'):
                try:
                    with open (os.path.join(input, file), 'r') as f:
                        data[file] = json.load(f)
                except Exception as e:
                    debug.log_error(f'Error loading JSON file {file}. {str(e)}')
                    return None
        return data
    

def sci_note_to_float(array, precision=2):
        
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
        debug.log_error(f'Invalid input. Expected a numpy array or a float, got {type(array)}')
        return None
        
def check_format(file):
    return file.split('.')[-1] in ['wav', 'aif', 'aiff', 'flac', 'ogg', 'mp3', 'json']


def nearest_power_of_2(x):
    """Find the nearest power of 2 less than or equal to x."""
    return 2**np.floor(np.log2(x))

def adjust_anal_res(args):
    """Adjust the analysis resolution based on the duration of the audio.
    Args:
        args: The arguments from the command line.
        Returns:
        n_fft: The number of samples in each frame.
        hop_size: The number of samples between successive frames.
        win_length: The number of samples in each window.
        n_bins: The number of frequency bins.
        n_mels: The number of mel bins.
    """
    if args.duration > 5.0:
        return args.n_fft, args.hop_size, args.window_length, args.n_bins, args.n_mels
    scale_factor = min(args.duration, 5.0) * 0.2 # scale based on a 5-second reference
    length = args.y.shape[-1]
    n_fft = int(nearest_power_of_2(args.n_fft * scale_factor))
    n_fft = min(n_fft, length)
    hop_size = max(8, int(n_fft * 0.25))
    # win_length = int(n_fft * 0.75) 
    win_length = int(384 * (scale_factor*0.5))
    n_bins = max(12, int(84 * scale_factor))
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
                        debug.log_error(f"Error decoding JSON from file: {file}")
                    except Exception as e:
                        debug.log_error(f"Error reading from file {file}. Error: {e}")
    else:
       raise ValueError("The data path must be a directory.")
    return data_dicts

#######################
# File management
#######################
def get_analysis_path():
    output_cache_dir = os.path.join(os.path.dirname(__file__), '..', 'cache', 'analysis')
    if not os.path.exists(output_cache_dir):
            os.makedirs(output_cache_dir)
    return output_cache_dir

def compute_curve(x, curve_type='exp'):
    """Compute a curve of length x.
    Args:
        x: The length of the curve in samples.
        curve_type: The type of curve to compute.
    Returns:
        The curve.
    """
    if curve_type == 'exp':
        fade_curve = np.linspace(0.0, 1.0, x) ** 2
    elif curve_type == 'log':
        fade_curve = np.sqrt(np.linspace(0.0, 1.0, x))
    elif curve_type == 'linear':
        fade_curve = np.linspace(0.0, 1.0, x)
    elif curve_type == 's_curve':
        t = np.linspace(0.0, np.pi / 2, x)
        fade_curve = np.sin(t)
    elif curve_type == 'hann':
        fade_curve = np.hanning(x) / 2 + 0.5  # or fade_curve = 0.5 * (1 - np.cos(np.pi * np.linspace(0.0, 1.0, fade_duration_samples)))
    elif curve_type == 'hamming':
        fade_curve = np.hamming(x)
        
    return fade_curve

def key_pressed(key):
    try:
        import msvcrt
        preseed = msvcrt.kbhit() and msvcrt.getch() == key
        return preseed
    except ImportError:
        import termios
        import sys
        import tty
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
            return ch == key
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
#######################
# progress
#######################
def progress_bar(current, total, message='Progress', barLength = 50):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))
    message = str(message)
    print(f'{message}: [{arrow}{spaces}] {percent:.2f} %', end='\r')
    if current == total:
        print('\n')

def processing_loop(stop_flag):
    while not stop_flag.is_set():
        for c in ['|', '/', '-', '\\']:
            print(f'\rProcessing {c}', end='')
            time.sleep(0.1)
stop_flag = threading.Event()
processing_thread = threading.Thread(target=processing_loop, args=(stop_flag,))

def on_process_start():
    processing_thread.start()

def on_process_end():
    stop_flag.set()
    processing_thread.join()
    
#######################
# ASCII Art
#######################
def print_ascii_art():
    print('''
    # +-------------------------------------+
    # |                                     |
    # |                                     |
    # |   _   _   _ ______ _     _ ______   |
    # |  | | | | | | |  | | |   | | |       |
    # |  | | | | | | |__| \ \   / / |----   |
    # |  |_|_|_|_|_/_|  |_|\_\_/_/|_|____   |
    # |   ____________ ______ ____________  |
    # |  | |   | |  | \ |  | | |     | |    |
    # |  | |   | |__| | |__| | |---- | |    |
    # |  |_|___|_|  \_\_|  |_|_|     |_|    |
    # |                                     |
    # |                                     |
    # +-------------------------------------+
    ''')
   
   
                                