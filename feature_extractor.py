#!/usr/bin/env python3.9
import librosa
import numpy as np
import os
import json
import argparse
import utils
from sklearn.decomposition import PCA

class Extractor:
    def __init__(self, args):
        self.args = args
        # self.args.ignore_descs = ['analysis', 'bit_rate', 'number_channels', 'sample_rate','codec', 'md5_encoded', 'tags', 'version', 'lossless']
        
    def extract(self, audio_file, output_file=None, output_dir=None, flatten_structure=True):
        audio_file = os.path.basename(audio_file)
        if not output_dir:
            file_dir = os.path.dirname(audio_file)
            output_dir = os.path.join(file_dir, 'analysis')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not output_file:
            output_file = os.path.join(output_dir, os.path.splitext(audio_file)[0]+'_analysis.json')
        errors = 0
        results = {}
        print("Analysing %s" % audio_file)
        
        if not output_dir:
            file_dir = os.path.dirname(audio_file)
            output_dir = os.path.join(file_dir, 'analysis')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        try:
            D = librosa.stft(self.args.y, n_fft=self.args.n_fft)
            S_P = np.abs(D)**2
            S_M = np.abs(D)
            C = librosa.cqt(y=self.args.y, sr=self.args.sample_rate, hop_length=self.args.hop_size, n_bins=self.args.n_bins)
            M = librosa.feature.melspectrogram(S=S_P, sr=self.args.sample_rate, n_fft=self.args.n_fft, hop_length=self.args.hop_size, n_mels=self.args.n_mels)
            
            # spectral features
            pca = PCA(n_components=12)  # the number of principal components to extract
            mel_spec_pca = pca.fit_transform(M.T).T 
            chroma_stft = librosa.feature.chroma_stft(S=S_P, sr=self.args.sample_rate, n_fft=self.args.n_fft, hop_length=self.args.hop_size)
            chroma_cqt = librosa.feature.chroma_cqt(C=C, sr=self.args.sample_rate, hop_length=self.args.hop_size)
            rms = librosa.feature.rms(S=S_M, hop_length=self.args.hop_size, frame_length=self.args.n_fft)
            spec_bw = librosa.feature.spectral_bandwidth(S=S_M, sr=self.args.sample_rate, hop_length=self.args.hop_size, n_fft=self.args.n_fft)
            spec_bw_delta = librosa.feature.delta(spec_bw, order=2)
            spec_cent = librosa.feature.spectral_centroid(S=S_M, sr=self.args.sample_rate, n_fft=self.args.n_fft, hop_length=self.args.hop_size)
            spec_cent_delta = librosa.feature.delta(spec_cent, order=2)
            spec_contrast = librosa.feature.spectral_contrast(S=S_M, sr=self.args.sample_rate, n_fft=self.args.n_fft, hop_length=self.args.hop_size)
            spec_contrast_delta = librosa.feature.delta(spec_contrast, order=2)
            rolloff = librosa.feature.spectral_rolloff(S=S_M, sr=self.args.sample_rate, n_fft=self.args.n_fft, hop_length=self.args.hop_size)
            rolloff_delta = librosa.feature.delta(rolloff, order=2)
            zcr = librosa.feature.zero_crossing_rate(self.args.y, hop_length=self.args.hop_size, frame_length=self.args.n_fft)
            zcr_delta = librosa.feature.delta(zcr, order=2)
            mfcc = librosa.feature.mfcc(S=librosa.power_to_db(M), sr=self.args.sample_rate)
            mfcc_delta = librosa.feature.delta(mfcc, order=2)
            poly_features = librosa.feature.poly_features(S=S_M, sr=self.args.sample_rate, hop_length=self.args.hop_size, n_fft=self.args.n_fft)
            tonnetz = librosa.feature.tonnetz(y=self.args.y, sr=self.args.sample_rate, chroma=chroma_cqt)
            
            # rythm features
            onset_env = librosa.onset.onset_strength(S=S_P, sr=self.args.sample_rate, lag=2)
            tempogram = librosa.feature.tempogram(y=self.args.y, sr=self.args.sample_rate, onset_envelope=onset_env, win_length=self.args.window_length, hop_length=self.args.hop_size)
            # tempo = librosa.feature.tempo(tg=tempogram, sr=self.args.sample_rate, hop_length=self.args.hop_size)
            # fourier_tempogram = librosa.feature.fourier_tempogram(y=self.args.y, sr=self.args.sample_rate, onset_envelope=onset_env, win_length=self.args.window_length, hop_length=self.args.hop_size)
            # tempo, beats = librosa.beat.beat_track(y=self.args.y, sr=self.args.sample_rate, hop_length=int(self.args.hop_size*0.25))
            tempogram_ratio = librosa.feature.tempogram_ratio(tg=tempogram, sr=self.args.sample_rate, hop_length=self.args.hop_size)
            
            features = {
                'mel_spec_stdv': mel_spec_pca, # 'mel_spec': M,
                'chroma_stft': chroma_stft,
                'rms': rms,
                'spec_bw': spec_bw,
                'spec_bw_delta': spec_bw_delta,
                'spec_cent': spec_cent,
                'spec_cent_delta': spec_cent_delta,
                'spec_contrast': spec_contrast,
                'spec_contrast_delta': spec_contrast_delta,
                'rolloff': rolloff,
                'rolloff_delta': rolloff_delta,
                'zcr': zcr,
                'zcr_delta': zcr_delta,
                'mfcc': mfcc,
                'mfcc_delta': mfcc_delta,
                'poly_features': poly_features,
                'tonnetz': tonnetz,
                'tempogram_ratio': tempogram_ratio
            }
            
            results = {}
            results['duration'] = self.args.duration
            
            for k, v in features.items():
                stdv = np.round(np.std(v, axis=1), 8)
                mean = np.round(np.mean(v, axis=1), 8)
                if np.iscomplexobj(stdv):
                    stdv = stdv.real
                if np.iscomplexobj(mean):
                    mean = mean.real
                results[k+'_stdv'] = stdv.tolist()
                results[k+'_mean'] = mean.tolist()
            
            # make everything JSON serializable
            for key in results:
                if type(results[key]) is np.ndarray:
                    results[key] = results[key].tolist()
            
            results = utils.flatten_dict(results) if flatten_structure else results
            
        except Exception as e:
            print("Error processing", audio_file, ":", str(e))
            return
                
        data = {'id': audio_file, 'stats': results}
        if os.path.isfile(output_file):
            with open(output_file, 'r') as outfile:
                old_data = json.load(outfile)
                # overwrite the stats with the new ones
                for key in data['stats']:
                    if key in old_data['stats'] and old_data['stats'][key] == data['stats'][key]:
                        print(utils.bcolors.YELLOW + "Warning: stats for " + key + " have not been updated as they are the same." + utils.bcolors.ENDC)
                        continue
                    old_data.setdefault('stats', {})[key] = data['stats'][key]
            with open(output_file, 'w') as outfile:
                json.dump(old_data, outfile, indent=4, sort_keys=True)
        else:
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=4, sort_keys=True)
    
    def main(self):
        
        self.extract(self.args.input_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = """
        Extracts audio features from a directory of audio files using librosa.
    """)
    parser.add_argument('-i', '--input-file', type=str, help='Input directory', required=True)
    parser.add_argument('-o', '--output-file', type=str, help='Output JSON file', required=False)
    parser.add_argument('-O', '--output-directory', type=str, help='Output directory to store descriptor files', required=False)
    parser.add_argument('-f', '--flaten-dict',  action='store_true', help='Flatten output dictionary so that all the nested dicts are combined into one', required=False, default=True)

    args = parser.parse_args()
    exrractor = Extractor(args)
    exrractor.main(args)