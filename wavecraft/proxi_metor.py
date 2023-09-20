#!/usr/bin/env python3.9

import asyncio, argparse, yaml, sys, os, shutil
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
import wavecraft.utils as utils

class ProxiMetor:
    def __init__(self, args):
        self.args = args
    
    def find_n_most_similar(self, identifier, df, metric=None, n=5, clss="stats"):
        """
        Find the indices of the n most similar files based on all metrics or a specific metric.
        Args:
            identifier: The ID of the file to compare.
            df: The DataFrame containing the data.
            metric: The metric to use for comparison.
            n: The number of similar files to retrieve.
            clss: The class to use for comparison, i.e., keys to nested dictionaries containing the metrics.
        Returns:
            A list of indices of the n most similar sounds.
        """
        # Standardize either specific metric or all metrics under a class
        descriptors_columns = [col for col in df.columns if clss in col]
        
        if metric:
            metric = clss + "_" + metric
            if metric not in df.columns:
                raise ValueError(f"The metric {metric} doesn't exist in the data.")
            scaler = StandardScaler()
            df[metric + "_standardized"] = scaler.fit_transform(df[[metric]])
            columns_to_compare = [metric + "_standardized"]
        else:
            scaler = StandardScaler()
            standardized_features = scaler.fit_transform(df[descriptors_columns])
            df[descriptors_columns] = standardized_features
            columns_to_compare = descriptors_columns
        
        sound_data = df[df["id"] == identifier].iloc[0]
        distances = []
        for index, row in df.iterrows():
            if row["id"] != identifier:
                dist = distance.euclidean(sound_data[columns_to_compare].values, row[columns_to_compare].values)
                distances.append((row["id"], dist))
        
        distances.sort(key=lambda x: x[1])
        return [item[0] for item in distances[:n]]

    def find_n_most_similar_weighted(self, identifier, df, ops):
        """
        Find the indices of the n most similar files based on all metrics or a specific metric.
        Args:
            identifier: The ID of the file to compare.
            df: The DataFrame containing the data.
            metric: The metric to use for comparison.
            n: The number of similar files to retrieve.
            clss: The class to use for comparison.
            ops: The options file containing the weights for each metric.
        Returns:
            A list of indices of the n most similar sounds.
        """
        clss = ops["class"]
        n = ops["n"]
        # Extract and standardize the metrics
        data_columns = [col for col in df.columns if clss in col]
        scaler = StandardScaler()
        standardized_features = scaler.fit_transform(df[data_columns])
        df[data_columns] = standardized_features
        columns_to_compare = data_columns

        # Set default weights
        weights = {col: 1 for col in data_columns}

        if 'weights' in ops:
            if ops['exclusive_weights'] is True:
                weights = {}  # Reset weights
            for key, value in ops['weights'].items():
                col_name = clss + "_" + key
                if col_name in data_columns:
                    weights[col_name] = value

        # Compute weighted Euclidean distance
        sound_data = df[df["id"] == identifier].iloc[0]
        if sound_data.isnull().values.any():
            raise ValueError("Invalid sound data")
        distances = []

        for index, row in df.iterrows():
            if row["id"] != identifier:
                weighted_diffs = [(sound_data[col] - row[col]) * weights.get(col, 1) for col in data_columns]
                dist = np.sqrt(sum(diff ** 2 for diff in weighted_diffs))
                distances.append((row["id"], dist))
        
        distances.sort(key=lambda x: x[1])
        print(distances)
        return [item[0] for item in distances[:n]]


    def find_n_most_similar_classifications(self, identifier, df, classification_category=None, n=5, clss="classifications"):
        """Find the indices of the n most similar files based on classifications."""
        
        if classification_category:
            # Extract specific classification columns
            columns_to_compare = [col for col in df.columns if clss in col and classification_category in col]
        else:
            # Extract all classification columns
            columns_to_compare = [col for col in df.columns if clss in col]
        
        sound_data = df[df["id"] == identifier]
        distances = []
        for index, row in df.iterrows():
            if row["id"] != identifier:
                dist = distance.euclidean(sound_data[columns_to_compare].values[0], row[columns_to_compare].values)
                distances.append((row["id"], dist))
        
        distances.sort(key=lambda x: x[1])
        return [item[0] for item in distances[:n]]

    def find_n_most_similar_for_a_file(self, used_files, id, df, metric=None, n=10, clss="stats", ops=None):
        """
        Find n most similar files for the given file which aren't in used_files.
        """
        df_copy = df.copy()
        df_copy = df_copy[~df_copy['id'].isin(used_files)]  # Exclude already used files before finding similar ones
        
        if(clss == "classifications"):
            return self.find_n_most_similar_classifications(id, df_copy, n=n, clss=clss)
        if ops:
            self.find_n_most_similar_weighted(id, df_copy, ops)
        else:
            return self.find_n_most_similar(id, df_copy, metric=metric, n=n, clss=clss)
            
    async def copy_similar_to_folders(self, base_path, data_path, file_id, similar_files):
        """
        Copy files of similar sounds to separate folders.
        Args:
            base_path: The base path to store the folders.
            data_path: The path to the data directory.
            file_id: The ID of the file to copy.
            similar_files: A list of IDs of similar files.
        Returns:
            None
        """
        if(len(similar_files) == 0):
            return
        # Create a directory for the sound_id
        target_folder = os.path.join(base_path, file_id)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        
        # create a directory for the analysis files
        analysis_folder = os.path.join(target_folder, "analysis")
        if not os.path.exists(analysis_folder):
            os.makedirs(analysis_folder)
            
        # This assumes that the source directory is one level above the data_path
        source_diectory = os.path.dirname(data_path)
        
        sound_files = similar_files + [file_id]
        for sound in sound_files:
            # Assuming each sound has an associated JSON file
            source_file_path = os.path.join(source_diectory, sound)
            source_file_without_extension = os.path.splitext(sound)[0]
            # also copy the analysis files
            analysis_file = source_file_without_extension+"_analysis.json"
            analysis_file_path = os.path.join(data_path, analysis_file)
            # Copy the file to the target directory
            if os.path.exists(source_file_path):
                # check if the file already exists in the target directory
                if not os.path.exists(os.path.join(target_folder, sound)):
                    print(f"Copying {source_file_without_extension} to {target_folder}")
                    shutil.copy2(source_file_path, target_folder)
                if not os.path.exists(os.path.join(analysis_folder, analysis_file)):
                    print(f"Copying {analysis_file_path} to {analysis_folder}")
                    shutil.copy2(analysis_file_path, analysis_folder)
            else:
                print(f"File {source_file_path} not found!")

            await asyncio.sleep(1)  # just to mimic some delay
        print(f"Copied similar sounds for {file_id} to {target_folder}.")

    async def process_batch(self, all_files, used_files, df, metric=None, n=5, clss="stats", id=None, ops=None):
        """Process a batch of sounds asynchronously."""
        if(id):
            primary_file = id
        else:
            primary_file = all_files.pop()
        similar_files = self.find_n_most_similar_for_a_file(used_files=used_files, 
                                                            id=primary_file, 
                                                            df=df, 
                                                            metric=metric, 
                                                            n=n, 
                                                            clss=clss, 
                                                            ops=ops)
        print(f"Found {len(similar_files)} similar files for {primary_file}.")
        await self.copy_similar_to_folders(base_path, data_path, primary_file, similar_files)
        used_files.update(similar_files)
        all_files.difference_update(used_files)
        
    def main(self):
        data_path = os.path.abspath(self.args.data_path)
        identifier_to_test = args.identifier
        base_path = self.args.base_path
        class_to_analyse = self.args.class_to_analyse
        metric_to_analyze = self.args.metric_to_analyze
        n = self.args.n
        use_options_file = self.args.ops
        ops = None
        n_max = args.n_max
        
        if(use_options_file):
            # Load the options yamle file from this directory
            ops_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "metric_ops.yaml")
            ops = yaml.load(open(ops_file), Loader=yaml.FullLoader)
            
        # Load the data
        data = utils.load_json(data_path)
        # Convert to DataFrame
        df = pd.json_normalize(data, sep="_")
        
        all_files = set(df["id"].tolist())
        if n_max == -1:
            n_max = len(all_files)
        used_files = set()
        loop = asyncio.get_event_loop()
        
        while all_files and len(used_files) < n_max:
            loop.run_until_complete(self.process_batch(all_files=all_files, 
                                                used_files=used_files, 
                                                df=df, 
                                                metric=metric_to_analyze, 
                                                n=n, 
                                                clss=class_to_analyse, 
                                                id=identifier_to_test, 
                                                ops=ops))
    

if(__name__ == "__main__"):
    
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('-d', '--data-path', type=str, required=True,
                        help='Path to the data directory')
    parser.add_argument('-id', '--identifier', type=str,
                        help='Identifier to test')
    parser.add_argument('-bp', '--base-path', type=str, default='./similar_files',
                        help='Base directory to store all similar file groups')
    parser.add_argument('-cls', '--class_to_analyse', type=str, default='stats',
                        help='Class to analyse')
    parser.add_argument('-m', '--metric-to-analyze', type=str, default=None,
                        help='Metric to analyze')
    parser.add_argument('-n', type=int, default=5,
                        help='Number of similar sounds to retrieve')
    parser.add_argument('-ops', action='store_true', default=False,
                        help='Use opetions file to fine tune the metric learning')
    parser.add_argument('-nm', '--n-max', type=int, default=-1, 
                        help='Max number of similar files to retrieve, Default: -1 (all)')

    args = parser.parse_args()
    


    
    
    
    
