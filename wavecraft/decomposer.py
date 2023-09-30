import sys
import os
import librosa
import asyncio
import traceback
import numpy as np
import sklearn.decomposition
from wavecraft.debug import Debug as debug
import soundfile as sf

class Decomposer:
    def __init__(self, input_file, method, n_components=4, render=False, render_path=None, output_file_harmonic=None, output_file_percussive=None, sample_rate=48000):
        self.input_file = input_file
        self.method = method
        self.n_components = n_components
        self.render = render
        self.render_path = render_path
        self.output_file_harmonic = output_file_harmonic
        self.output_file_percussive = output_file_percussive
        self.sample_rate = sample_rate

    async def decompose(self):
        y, sr = librosa.load(self.input_file, sr=self.sample_rate)

        if self.render_path is None and self.render:
            self.render_path = os.path.join(os.path.dirname(self.input_file), 'components')

        if self.method == 'decompose':
            comps, acts = await self._decompose(y, n_components=self.n_components, render_path=self.render_path)
            if not self.render:
                debug.log_info(f'Decomposed the signal into {self.n_components} components.')
                debug.log_info(f'Render not requested. Exiting...')
                return

            debug.log_info(f'Decomposed the signal into {self.n_components} components.')
            debug.log_info(f'Rendering components to {self.render_path}...')
            self.render_components(comps, acts, sr, self.render_path)

        elif self.method == 'hpss':
            y_harmonic, y_percussive = await self._decompose_hpss(y)
            if not self.render:
                debug.log_info(f'Decomposed the signal into harmonic and percussive components.')
                debug.log_info(f'Render not requested. Exiting...')
                return
            debug.log_info(f'Decomposed the signal into harmonic and percussive components.')
            debug.log_info(f'Rendering components to {self.render_path}...')
            self.render_hpss(y_harmonic, y_percussive, self.render_path)

        elif self.method == 'sklearn':
            scomps, sacts = await self._decompose_sk(y, n_components=self.n_components)
            if not self.render:
                debug.log_info(f'Decomposed the signal into {self.n_components} components.')
                debug.log_info(f'Render not requested. Exiting...')
                return
            debug.log_info(f'Decomposed the signal into {self.n_components} components.')
            debug.log_info(f'Rendering components to {self.render_path}...')
            self.render_components(scomps, sacts, sr, self.render_path)

    async def _decompose(self, y, n_components=4, spectogram=None, n_fft=1024, hop_length=512, render_path=None):
        if spectogram is None:
            spectogram = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        S, phase = librosa.magphase(spectogram)
        comps, acts = librosa.decompose.decompose(S, n_components=n_components, sort=True)
        return comps, acts

    async def _decompose_hpss(self, y, n_fft=1024, hop_length=512):
        s = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        H, P = librosa.decompose.hpss(s)
        y_harmonic = librosa.istft(H, length=len(y))
        y_percussive = librosa.istft(P, length=len(y))
        return y_harmonic, y_percussive

    async def _decompose_sk(self, y, n_components, n_fft=1024, hop_length=512):
        S = np.abs(librosa.stft(y))
        T = sklearn.decomposition.MiniBatchDictionaryLearning(n_components=16)
        scomps, sacts = librosa.decompose.decompose(S, transformer=T, sort=True)
        return scomps, sacts
    
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

    def run(self):
        try:
            asyncio.run(self.decompose())
        except Exception as e:
            debug.log_info(f'Error: {type(e).__name__}! {e}')
            traceback.debug.log_info_exc()
            sys.exit(1)
            
