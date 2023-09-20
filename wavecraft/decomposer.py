import sys
import os
import argparse
import librosa
import asyncio
import traceback
import numpy as np
import sklearn.decomposition
import utils

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
                print(f'{utils.bcolors.GREEN}Decomposed the signal into {self.n_components} components.{utils.bcolors.ENDC}')
                print(f'{utils.bcolors.YELLOW}Render not requested. Exiting...{utils.bcolors.ENDC}')
                return

            print(f'{utils.bcolors.GREEN}Decomposed the signal into {self.n_components} components.{utils.bcolors.ENDC}')
            print(f'{utils.bcolors.YELLOW}Rendering components to {self.render_path}...{utils.bcolors.ENDC}')
            utils.render_components(comps, acts, sr, self.render_path)

        elif self.method == 'hpss':
            y_harmonic, y_percussive = await self._decompose_hpss(y)
            if not self.render:
                print(f'{utils.bcolors.GREEN}Decomposed the signal into harmonic and percussive components.{utils.bcolors.ENDC}')
                print(f'{utils.bcolors.YELLOW}Render not requested. Exiting...{utils.bcolors.ENDC}')
                return
            print(f'{utils.bcolors.GREEN}Decomposed the signal into harmonic and percussive components.{utils.bcolors.ENDC}')
            print(f'{utils.bcolors.YELLOW}Rendering components to {self.render_path}...{utils.bcolors.ENDC}')
            utils.render_hpss(y_harmonic, y_percussive, self.render_path)

        elif self.method == 'sklearn':
            scomps, sacts = await self._decompose_sk(y, n_components=self.n_components)
            if not self.render:
                print(f'{utils.bcolors.GREEN}Decomposed the signal into {self.n_components} components.{utils.bcolors.ENDC}')
                print(f'{utils.bcolors.YELLOW}Render not requested. Exiting...{utils.bcolors.ENDC}')
                return
            print(f'{utils.bcolors.GREEN}Decomposed the signal into {self.n_components} components.{utils.bcolors.ENDC}')
            print(f'{utils.bcolors.YELLOW}Rendering components to {self.render_path}...{utils.bcolors.ENDC}')
            utils.render_components(scomps, sacts, sr, self.render_path)

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

    def run(self):
        try:
            asyncio.run(self.decompose())
        except Exception as e:
            print(f'Error: {type(e).__name__}! {e}')
            traceback.print_exc()
            sys.exit(1)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Decompose an audio file using different methods')
    parser.add_argument('-i', '--input_file', type=str, help='path to the input audio file', required=True)
    parser.add_argument('-m', '--method', type=str, choices=['decompose', 'hpss', 'sklearn'], help='method to use for decomposition', required=True)
    parser.add_argument('-n', '--n_components', type=int, default=4, help='number of components to use for decomposition')
    parser.add_argument('-r', '--render', action='store_true', dest='render', help='render the decomposed signal')
    parser.add_argument('-p', '--render_path', type=str, default=None, help='path to render the decomposed signal. Defaults to the audio file directory')
    parser.add_argument('-o_h', '--output_file_harmonic', type=str, default=None, help='path to output the harmonic component')
    parser.add_argument('-o_p', '--output_file_percussive', type=str, default=None, help='path to output the percussive component')
    parser.add_argument('-s', '--sample_rate', type=int, default=48000, help='sample rate of the input audio file')
    args = parser.parse_args()

    decomposer = Decomposer(args.input_file, args.method, args.n_components, args.render, args.render_path, args.output_file_harmonic, args.output_file_percussive, args.sample_rate)
    decomposer.run()