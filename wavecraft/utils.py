
import threading
import json
import os
import time
import librosa
import numpy as np
import sounddevice as sd
from .debug import Debug as debug, colors


#######################
# Spectrogram
#######################

def compute_spectrogram(y, sr, spec_type, n_fft, hop_length, n_mels, fmin):
    print(n_fft, hop_length, n_mels, fmin)
    if spec_type == 'mel':
        return librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    elif spec_type == 'cqt':
        return np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length, fmin=fmin))
    elif spec_type == 'stft':
        return np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    elif spec_type == 'cqt_chroma':
        return librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length, fmin=fmin)
    else:
        raise ValueError(f"Unsupported spec_type: {spec_type}")

#######################
# Maths
#######################    
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
    
def nearest_power_of_2(x):
    """Find the nearest power of 2 less than or equal to x."""
    return 2**np.floor(np.log2(x))

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

def adjust_anal_res(args):
    """
    Adjust the analysis resolution based on the duration and sample rate of the audio.
    
    Args:
        args: The arguments from the command line.
    
    Returns:
        n_fft: The number of samples in each frame.
        hop_size: The number of samples between successive frames.
        win_length: The number of samples in each window.
        n_bins: The number of frequency bins.
        n_mels: The number of mel bins.
    """
    standard_rate = 22050  # Standard sample rate (e.g., 44.1 kHz)
    sample_rate_factor = args.sample_rate / standard_rate

    # Scale factor based on duration and sample rate
    duration_scale = min(args.duration, 5.0) / 5.0
    scale_factor = duration_scale * sample_rate_factor

    length = args.y.shape[-1]
    n_fft = int(nearest_power_of_2(args.n_fft * scale_factor))
    n_fft = min(n_fft, args.n_fft)  # Ensure n_fft <= length
    n_fft = max(128, n_fft)  # Ensure n_fft >= 8
    hop_size = max(32, int(n_fft * 0.25))
    
    # Adjust window length and other parameters based on scale factor
    win_length = max(48, int(384 * (scale_factor * 0.25)))
    win_length = min(win_length, n_fft)  # Ensure win_length <= n_fft
    n_bins = max(12, int(84 * scale_factor)) 
    n_bins = min(n_bins, n_fft // 2 + 1)
    n_mels = max(12, int(128 * scale_factor))
    
    # print(n_fft, hop_size, win_length, n_bins, n_mels, args.sample_rate, length)
    return n_fft, hop_size, win_length, n_bins, n_mels


#######################
# Preview
#######################

def preview_audio(y, sr):
    """
    A standalone function to play audio data and wait for user input.
    
    Parameters:
    - y: The filtered audio data.
    - sample_rate: Sample rate for audio playback.
    - logger: A logging object/module with log_any, log_info, log_warning, and log_error methods.
    
    Returns:
    - True if user confirms, else False.
    """
    print()
    debug.log_any('Results...', 'preview')
    y = basic_process(y, sr)
    sd.play(y, samplerate=sr)
        
    while True:
        confirmation = input(f"\n{colors.GREEN}Do you want to render the results?{colors.ENDC}\n\n1) Confirm\n2) Replay preview\n3) Exit\n")
        
        if confirmation.lower() == '1':
            debug.log_info("Result confirmed")
            sd.stop()
            return True
        elif confirmation.lower() == '2':
            debug.log_info("Replaying")
            sd.play(y, samplerate=sr)
            sd.wait()
        elif confirmation.lower() == '3':
            debug.log_warning("Exiting without confirmation")
            return False
        else:
            debug.log_error("Invalid input! Choose one of the options below.", False)
            continue

def basic_process(y, sr, normalise=True, highPassFilter=True):
    from .processor import Processor
    processor = Processor()
    y = processor.filter(y, sr, 40)
    y = processor.normalise_audio(y, sr, -3)
    y = processor.fade_io(y, sr)
    return y

def finish_timer(condition):
    condition = True

#######################
# File management
#######################   
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
    
        
def check_format(file):
    return file.split('.')[-1] in ['wav', 'aif', 'aiff', 'flac', 'ogg', 'mp3', 'json']


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

def get_analysis_path():
    path = os.path.join(os.path.dirname(__file__), '..', 'cache', 'analysis')
    if not os.path.exists(path):
            os.makedirs(path)
    return path

def get_output_path():
    path = os.path.join(os.path.dirname(__file__), '..', 'cache', 'output')
    if not os.path.exists(path):
            os.makedirs(path)
    return path


#######################
# Input
#######################
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
# UI
#######################
def progress_bar(current, total, message='Progress', barLength = 50):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))
    message = str(message)
    print(f'{message}: [{arrow}{spaces}] {percent:.2f} %', end='\r')
    if current == total:
        print('\n')
        
def progress(operation):
    t=0
    while t <= 1.01:
        progress_bar(t, 1, message=operation)
        t+=0.01
        time.sleep(0.004)

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
    
def print_end():
    print('''
                    o-o  o 
                   |    | 
     o-o o-o  oo  -O-  -o-
    |    |   / |   |    | 
     o-o o   o-o-  o    o                                                                                                          
    ''')
    
def print_seperator():
    print('''
    +-------------------------------------+
    ''')
