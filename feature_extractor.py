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
        errors = {}
        results = {}
        
        
        if not output_dir:
            file_dir = os.path.dirname(audio_file)
            output_dir = os.path.join(file_dir, 'analysis')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        try:
            D = librosa.stft(self.args.y, n_fft=self.args.n_fft)
            S_P = np.abs(D)**2
            S_M = librosa.magphase(D)[0]
            rn_mels = 12
            n_mels = max(rn_mels, self.args.n_mels)
            M = librosa.feature.melspectrogram(S=S_P, sr=self.args.sample_rate, n_fft=self.args.n_fft, hop_length=self.args.hop_size, n_mels=n_mels)
            
            # spectral features
            chroma_stft = librosa.feature.chroma_stft(S=S_P, sr=self.args.sample_rate, n_fft=self.args.n_fft, hop_length=self.args.hop_size)
            rms = librosa.feature.rms(S=S_M, hop_length=self.args.hop_size, frame_length=self.args.n_fft)
            
            spec_bw = librosa.feature.spectral_bandwidth(S=S_M, sr=self.args.sample_rate, hop_length=self.args.hop_size, n_fft=self.args.n_fft)
            spec_bw_delta = librosa.feature.delta(spec_bw, order=2)
            
            spec_cent = librosa.feature.spectral_centroid(S=S_M, sr=self.args.sample_rate, n_fft=self.args.n_fft, hop_length=self.args.hop_size)
            spec_cent_delta = librosa.feature.delta(spec_cent, order=2)
            
            spec_contrast = librosa.feature.spectral_contrast(y=self.args.y, sr=self.args.sample_rate, hop_length=self.args.hop_size)
            spec_contrast_delta = librosa.feature.delta(spec_contrast, order=2)
            
            spec_flatness = librosa.feature.spectral_flatness(S=S_M, n_fft=self.args.n_fft, hop_length=self.args.hop_size)
            spec_flatness_delta = librosa.feature.delta(spec_flatness, order=2)
            
            rolloff = librosa.feature.spectral_rolloff(S=S_M, sr=self.args.sample_rate, n_fft=self.args.n_fft, hop_length=self.args.hop_size)
            rolloff_delta = librosa.feature.delta(rolloff, order=2)
            
            zcr = librosa.feature.zero_crossing_rate(self.args.y, hop_length=self.args.hop_size, frame_length=self.args.n_fft)
            zcr_delta = librosa.feature.delta(zcr, order=2)
            
            mfcc = librosa.feature.mfcc(S=librosa.power_to_db(M), sr=self.args.sample_rate)
            mfcc_delta = librosa.feature.delta(mfcc, order=2)
            
            poly_features = librosa.feature.poly_features(S=S_M, sr=self.args.sample_rate, hop_length=self.args.hop_size, n_fft=self.args.n_fft)
            tonnetz = librosa.feature.tonnetz(y=self.args.y, sr=self.args.sample_rate, chroma=chroma_stft)

            # rythm features
            onset_env = librosa.onset.onset_strength(S=S_P, sr=self.args.sample_rate, lag=2)
            tempogram = librosa.feature.tempogram(y=self.args.y, sr=self.args.sample_rate, onset_envelope=onset_env, win_length=self.args.window_length, hop_length=self.args.hop_size)
            fourier_tempogram = librosa.feature.fourier_tempogram(y=self.args.y, sr=self.args.sample_rate, onset_envelope=onset_env, win_length=self.args.window_length)
            tempogram_ratio = librosa.feature.tempogram_ratio(tg=tempogram, sr=self.args.sample_rate, hop_length=self.args.hop_size)
            
            n_mffc = 20
            if mfcc.shape[0] < n_mffc:
                print(utils.bcolors.YELLOW+"Warning: padding mfcc with zeros to match the number of rows of the PCA matrix"+utils.bcolors.ENDC)
                pad_amount = n_mffc - mfcc.shape[0]
                mfcc = np.pad(mfcc, ((0, pad_amount), (0, 0)), mode='constant')
            
            fourier_tempogram = np.real(fourier_tempogram)
            n_ftempo = 20
            if fourier_tempogram.shape[0] < n_ftempo:
                print(utils.bcolors.YELLOW+"Warning: padding fourier tempogram with zeros to match the number of rows of the PCA matrix"+utils.bcolors.ENDC)
                pad_amount = n_ftempo - fourier_tempogram.shape[0]
                fourier_tempogram = np.pad(fourier_tempogram, ((0, pad_amount), (0, 0)), mode='constant')
            pca_ftempo = PCA(n_components=n_ftempo)
            fourier_tempogram = pca_ftempo.fit_transform(fourier_tempogram.T).T
            
            if M.shape[0] < rn_mels:
                print(utils.bcolors.YELLOW+"Warning: padding mel spectrogram with zeros to match the number of rows of the PCA matrix"+utils.bcolors.ENDC)
                pad_amount = rn_mels - M.shape[0]
                M = np.pad(M, ((0, pad_amount), (0, 0)), mode='constant')
            pca_M = PCA(n_components=rn_mels)
            M_pca = pca_M.fit_transform(M.T).T
            
            features = {
                'mel_spec_stdv': M_pca, # 'mel_spec': M,
                'chroma_stft': chroma_stft,
                'rms': rms,
                'spec_bw': spec_bw,
                'spec_bw_delta': spec_bw_delta,
                'spec_cent': spec_cent,
                'spec_cent_delta': spec_cent_delta,
                'spec_contrast': spec_contrast,
                'spec_contrast_delta': spec_contrast_delta, 
                'spec_flatness': spec_flatness, 
                'spec_flatness_delta': spec_flatness_delta,
                'rolloff': rolloff,
                'rolloff_delta': rolloff_delta,
                'zcr': zcr,
                'zcr_delta': zcr_delta,
                'mfcc': mfcc,
                'mfcc_delta': mfcc_delta,
                'poly_features': poly_features,
                'tempogram_ratio': tempogram_ratio,
                'fourier_tempogram': fourier_tempogram,
                'tonnetz': tonnetz,
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
            print(utils.bcolors.RED+"Error processing", audio_file, ":", str(e)+utils.bcolors.ENDC)
            errors[audio_file] = str(e)
            return errors
                
        data = {'id': audio_file, 'stats': results}
        if os.path.isfile(output_file):
            with open(output_file, 'r') as outfile:
                old_data = json.load(outfile)
                # overwrite the stats with the new ones
                for key in data['stats']:
                    if key in old_data['stats'] and old_data['stats'][key] == data['stats'][key]:
                        # print(utils.bcolors.YELLOW + "Warning: stats for " + key + " have not been updated as they are the same." + utils.bcolors.ENDC)
                        continue
                    old_data.setdefault('stats', {})[key] = data['stats'][key]
            with open(output_file, 'w') as outfile:
                json.dump(old_data, outfile, indent=4, sort_keys=True)
        else:
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=4, sort_keys=True)
    
    def main(self):
        self.extract(self.args.input_file)

