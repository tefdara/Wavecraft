import sys
import os
import librosa
import asyncio
import traceback
import numpy as np
import sklearn.decomposition
from .debug import Debug as debug
import soundfile as sf
from . import utils

class Decomposer:
    def __init__(self, args, render = False):
        self.args = args
        self.render = render
        self.input_file_name = os.path.splitext(os.path.basename(args.input))[0]
        self.output_path = os.path.join(os.path.dirname(args.input), self.input_file_name + '_decomposed')
        self.output_file_harmonic = os.path.join(self.output_path, args.input + '_harmonic.wav')
        self.output_file_percussive = os.path.join(self.output_path, args.input + '_percussive.wav')
        from .processor import Processor
        self.processor = Processor(self.args)
        
        if args.sklearn and args.nn_filter:
            debug.log_error(f'Cannot use sklearn and nn_filter together.')
        if args.sklearn or args.nn_filter:
            if args.source_separation is not None:
                debug.log_error(f'Cannot use sklearn with source separation.')
            if args.sklearn:self.method = 'sklearn'
            if args.nn_filter:
                self.method = 'nn_filter'
        if not args.sklearn and not args.nn_filter:
            if args.source_separation is None:
                self.method = 'decompose'
            else:
                self.method = 'hpss'    
        
    async def main(self):
        if self.method == 'decompose':
            comps, acts, phase = await self._decompose_n(self.args.y, n_components=self.args.n_components)
            debug.log_info(f'Decomposed the signal into <{self.args.n_components}> components.')
            if not self.render:
                debug.log_info(f'Render not requested. Returning components...')
                return comps, acts

            debug.log_info(f'Rendering components to {self.output_path}...')
            self.render_components(comps, acts, self.args.n_components, phase)
            
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
            scomps, sacts, sphase = await self._decompose_sk(self.args.y, n_components=self.args.n_components)
            debug.log_info(f'Decomposed the signal into <{self.args.n_components} components>, using <sklearn>.')
            if not self.render:
                debug.log_info(f'Render not requested. Returning components...')
                return
            debug.log_info(f'Rendering components to {self.output_path}...')
            self.render_components(scomps, sacts, self.args.n_components, sphase)
        
        elif self.method == 'nn_filter':
            debug.log_info(f'Filtering the signal using <nn_filter>...')
            await self._decompose_nn_filter(self.args.y)

    async def _decompose_n(self, y, n_components=4, spectogram=None):
        if spectogram is None:
            spectogram = librosa.stft(y, n_fft=self.args.n_fft, hop_length=self.args.hop_size)
        S, phase = librosa.magphase(spectogram)
        comps, acts = librosa.decompose.decompose(S, n_components=n_components, sort=True, max_iter=1000)
        return comps, acts, phase

    async def _decompose_hpss(self, y):
        s = librosa.stft(y, n_fft=self.args.n_fft, hop_length=self.args.hop_size)
        H, P = librosa.decompose.hpss(s)
        y_harmonic = librosa.istft(H, length=len(y))
        y_percussive = librosa.istft(P, length=len(y))
        return y_harmonic, y_percussive

    async def _decompose_sk(self, y, n_components):
        S = np.abs(librosa.stft(y))
        T = sklearn.decomposition.MiniBatchDictionaryLearning(n_components=n_components, max_iter=1000)
        scomps, sacts = librosa.decompose.decompose(S, transformer=T, sort=True)
        sphase = np.exp(1.j * np.angle(librosa.stft(y)))
        return scomps, sacts, sphase
    
    async def _decompose_nn_filter(self, y):
        if self.args.spectogram is None:
            S = librosa.feature.melspectrogram(y=y, sr=self.args.sample_rate, n_mels=128)
        else:
            S = utils.compute_spectrogram(y, self.args.sample_rate, self.args.spectogram, self.args.n_fft, self.args.hop_size, self.args.n_mels, self.args.fmin)
            
        # S_db = librosa.power_to_db(S, ref=np.max)
        S_filtered = librosa.decompose.nn_filter(S, aggregate=np.median, metric='cosine')
        y_filtered = librosa.feature.inverse.mel_to_audio(S_filtered, sr=self.args.sample_rate)
        
        if utils.preview_audio(y_filtered, self.args.sample_rate) == True:
            debug.log_info(f'Rendering <filtered> signal to {self.output_path}...')
            self.render_core(y_filtered, self.args.sample_rate, os.path.join(self.output_path, self.input_file_name + '_filtered.wav'))
        
        
    def render_components(self, components, activations, n_components, phase):
    
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        
        output_file = os.path.join(self.output_path, self.input_file_name + '_component_')
        
        for i in range(n_components):
            # Combine the component with the phase
            component_spectrogram = components[:, i:i+1] @ activations[i:i+1, :]
            component_complex = component_spectrogram * phase
            # Get the audio signal back using inverse STFT
            y_comp = librosa.istft(component_complex, hop_length=self.args.hop_size)
            file = output_file + str(i+1) + ".wav"
            debug.log_info(f'Rendering <component {i+1}> to {file}...')
            self.render_core(y_comp, self.args.sample_rate, file)

    def render_hpss(self, y, sr, render_path, type='harmonic'):
        self.render_core(y, sr, os.path.join(render_path, self.input_file_name + '_' + type + '.wav'))
        
    def render_core(self, y, sr, output_file, normalise=True, highPassFilter=True):
        dir = os.path.dirname(output_file)
        if not os.path.exists(dir):
            os.makedirs(dir)
            
        y = self.process(y, sr, normalise, highPassFilter)
        
        sf.write(output_file, y, sr)
        
    def process(self, y, sr, normalise=True, highPassFilter=True):
        if highPassFilter:
            if self.args.filter_frequency == 40:
                self.args.filter_frequency = 50
            y = self.processor.filter(y, sr, self.args.filter_frequency)
       
        if normalise:
            if self.args.normalisation_level == -3:
                self.args.normalisation_level = -5
            y = self.processor.normalise_audio(y, sr, self.args.normalisation_level)
            
        self.processor.fade_io(y, sr, fade_in=self.args.fade_in, fade_out=self.args.fade_out)
        return y

    def run(self):
        try:
            asyncio.run(self.main())
        except Exception as e:
            debug.log_info(f'Error: {type(e).__name__}! {e}')
            traceback.debug.log_info_exc()
            sys.exit(1)
            
