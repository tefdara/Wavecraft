import os, librosa, soundfile as sf, numpy as np

# Define color codes for print messages
class bcolors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'

def render_components(components, activations, n_components, phase, render_path, sr=48000, hop_length=512):
    
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

def render_hpss(y_harmonic, y_percussive, render_path, sr=48000):
    if not os.path.exists(render_path):
        os.makedirs(render_path)
        
    sf.write(os.path.join(render_path, 'harmonic.wav'), y_harmonic, sr)
    sf.write(os.path.join(render_path, 'percussive.wav'), y_percussive, sr)    
        
def apply_fadeout(audio, sr, fade_duration=50, curve_type='exp'):
    # convert fade duration to samples
    fade_duration_samples = int(fade_duration * sr / 1000)
    
    # If fade_duration_samples is larger than the segment, we should adjust it
    fade_duration_samples = min(fade_duration_samples, len(audio))
    
    # switch between different curve types
    if curve_type == 'exp':
        fade_curve = np.linspace(1.0, 0.0, fade_duration_samples) ** 2
    elif curve_type == 'log':
        fade_curve = np.sqrt(np.linspace(1.0, 0.0, fade_duration_samples))
    elif curve_type == 'linear':
        fade_curve = np.linspace(1.0, 0.0, fade_duration_samples)
    elif curve_type == 's_curve':
        t = np.linspace(0.0, np.pi / 2, fade_duration_samples)
        fade_curve = np.sin(t)
    elif curve_type == 'hann':
        fade_curve = np.hanning(fade_duration_samples) # or fade_curve = 0.5 * (1 - np.cos(np.pi * np.linspace(0.0, 1.0, fade_duration_samples)))
 
    
    # compute fade out curve
    fade_curve = np.linspace(1.0, 0.0, fade_duration_samples)
    
    # apply the curve only to the last part of the audio equivalent to fade_duration
    audio[-fade_duration_samples:] *= fade_curve
    
    return audio