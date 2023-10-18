import sys
import os
import librosa
import asyncio
import traceback
import numpy as np
import sklearn.decomposition
from .debug import Debug as debug
import soundfile as sf

class Decomposer:
    def __init__(self, args, render = False):
        self.args = args
        self.render = render
        self.output_path = os.path.splitext(args.input)[0] + '_decomposed'
        self.output_file_harmonic = os.path.join(self.output_path, 'harmonic.wav')
        self.output_file_percussive = os.path.join(self.output_path, 'percussive.wav')
        if args.source_separation is None:
            self.method = 'decompose'
        else:
            self.method = 'hpss'    
        
    async def main(self):

        # if self.output_path is None and self.render:
        #     self.output_path = os.path.join(os.path.dirname(self.input_file), 'components')

        if self.method == 'decompose':
            comps, acts = await self._decompose_n(self.args.y, n_components=self.args.n_components, render_path=self.output_path)
            debug.log_info(f'Decomposed the signal into {self.n_components} components.')
            if not self.render:
                debug.log_info(f'Render not requested. Returning components...')
                return comps, acts

            debug.log_info(f'Rendering components to {self.output_path}...')
            self.render_components(comps, acts, self.args.sample_rate, self.output_path)
            return comps, acts

        elif self.method == 'hpss':
            y_harmonic, y_percussive = await self._decompose_hpss(self.args.y)
            debug.log_info(f'<Decomposed> the signal into harmonic and percussive components.')
            
            if not self.render:
                debug.log_info(f'Render not requested. Returning components...')
            
            else:
                if self.args.source_separation == 'harmonic':
                    debug.log_info(f'<Rendering harmonic> component to {self.output_file_harmonic}...')
                    self.render_hpss(y_harmonic, self.args.sample_rate, self.output_path, type='harmonic')
                    
                elif self.args.source_separation == 'percussive':
                    debug.log_info(f'<Rendering percussive> component to {self.output_file_percussive}...')
                    self.render_hpss(y_percussive, self.args.sample_rate, self.output_path, type='percussive')
                    
                elif self.args.source_separation == 'hp':
                    debug.log_info(f'<Rendering harmonic> component to {self.output_file_harmonic}...')
                    self.render_hpss(y_harmonic, self.args.sample_rate, self.output_path, type='harmonic')
                    debug.log_info(f'<Rendering percussive> component to {self.output_file_percussive}...')
                    self.render_hpss(y_percussive, self.args.sample_rate, self.output_path, type='percussive')
                    
                else:
                    debug.log_error(f'Invalid HPSS type: {self.args.hpss}')
            
            return y_harmonic, y_percussive
        
        elif self.method == 'sklearn':
            scomps, sacts = await self._decompose_sk(self.args.y, n_components=self.n_components)
            if not self.render:
                debug.log_info(f'Decomposed the signal into {self.n_components} components.')
                debug.log_info(f'Render not requested. Exiting...')
                return
            debug.log_info(f'Decomposed the signal into {self.n_components} components.')
            debug.log_info(f'Rendering components to {self.output_path}...')
            self.render_components(scomps, sacts, self.args.sample_rate, self.output_path)

    async def _decompose_n(self, y, n_components=4, spectogram=None, n_fft=1024, hop_length=512, render_path=None):
        if spectogram is None:
            spectogram = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        S, phase = librosa.magphase(spectogram)
        comps, acts = librosa.decompose.decompose(S, n_components=n_components, sort=True)
        return comps, acts

    async def _decompose_hpss(self, y):
        s = librosa.stft(y, n_fft=self.args.n_fft, hop_length=self.args.hop_size)
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

    def render_hpss(self, y, sr, render_path, type='harmonic'):
        if not os.path.exists(render_path):
            os.makedirs(render_path)
        sf.write(os.path.join(render_path, type+'.wav'), y, sr)

    def run(self):
        try:
            asyncio.run(self.main())
        except Exception as e:
            debug.log_info(f'Error: {type(e).__name__}! {e}')
            traceback.debug.log_info_exc()
            sys.exit(1)
            
