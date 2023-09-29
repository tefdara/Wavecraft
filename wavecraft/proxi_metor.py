#!/usr/bin/env python3.9

import asyncio, yaml, sys, os, shutil
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
import wavecraft.utils as utils
from wavecraft.debug import Debug as debug

class ProxiMetor:
    def __init__(self, args):
        self.args = args
        if os.path.isdir(self.args.input):
            self.audio_path = os.path.abspath(self.args.input)
            self.data_path = os.path.join(self.audio_path, 'analysis')
            if not os.path.exists(self.data_path):
                self.data_path = utils.get_analysis_path()
        else:
            utils.error_logger.error('Invalid input! Please provide a directory for proximity analysis.')
            sys.exit(1)
            
        self.ops = None
        self.base_path = './similar_files'
        if(self.args.ops):
            # Load the options yamle file from this directory
            ops_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "metric_ops.yaml")
            self.ops = yaml.load(open(ops_file), Loader=yaml.FullLoader)
    
    def expand_nested_columns(self, df):
        new_columns = []
        drop_columns = []
        
        # Process each column
        for col in df.columns:
            if isinstance(df[col].iloc[0], list):
                # Create separate columns for each list item
                expanded_col = pd.DataFrame(df[col].tolist(), columns=[f"{col}_{i}" for i in range(len(df[col].iloc[0]))])
                new_columns.append(expanded_col)
                drop_columns.append(col)

        # Concatenate original dataframe with the new columns
        df = pd.concat([df.drop(columns=drop_columns)] + new_columns, axis=1)
        
        return df
            
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
        df = self.expand_nested_columns(df)
        scaler = StandardScaler()
            
        # Standardize either specific metric or all metrics under a class
        if metric:
            if metric.endswith('_'):  # Wildcard matching
                print(f'{utils.colors.MAGENTA}Using wildcard matching for {metric} {utils.colors.ENDC}')
                prefix = clss + "_" + metric[:-1] # Remove the wildcard
                columns_to_compare = [col for col in df.columns if col.startswith(prefix)]
                df[columns_to_compare] = scaler.fit_transform(df[columns_to_compare])
            else:
                print(f'{utils.colors.MAGENTA}Using exact matching for {metric} {utils.colors.ENDC}')
                metric = clss + "_" + metric
                if metric not in df.columns:
                    raise ValueError(f"The metric {metric} doesn't exist in the data.")
                df[metric + "_standardized"] = scaler.fit_transform(df[[metric]])
                columns_to_compare = [metric + "_standardized"]
        else:
            print(f'{utils.colors.MAGENTA}Using all metrics for {clss} {utils.colors.ENDC}')
            descriptors_columns = [col for col in df.columns if clss in col]
            standardized_features = scaler.fit_transform(df[descriptors_columns])
            df[descriptors_columns] = standardized_features
            columns_to_compare = descriptors_columns
        
        # Filter by the given identifier
        sound_data = df[df["id"] == identifier].iloc[0]
        
        # Compute distances for all rows not equal to the given identifier
        distances = df[df["id"] != identifier].apply(
            lambda row: (row["id"], distance.euclidean(sound_data[columns_to_compare].values, row[columns_to_compare].values)), 
            axis=1
        ).tolist()
        
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

    def find_all_based_on_metric(self, identifier, df, metric, metric_range, clss="stats"):
        scaler = StandardScaler()
        if identifier is None:
            identifier = df[(df[clss + "_" + metric] >= metric_range[0]) & (df[clss + "_" + metric] <= metric_range[1])].iloc[0]["id"]
        # columns_to_compare = self.get_metric_columns(df, metric, scaler, clss)
        # sound_data = df[df["id"] == identifier].iloc[0]
        # distances = df[df["id"] != identifier].apply(
        #     lambda row: (row["id"], distance.euclidean(sound_data[columns_to_compare].values, row[columns_to_compare].values)), 
        #     axis=1
        # ).tolist()
        # distances.sort(key=lambda x: x[1])
        return df[(df[clss + "_" + metric] >= metric_range[0]) & (df[clss + "_" + metric] <= metric_range[1])]["id"].tolist() 
        
      
        
        print(metric_range)
        return [item[0] for item in distances if item[1] >= metric_range[0] and item[1] <= metric_range[1]]
    
    def get_metric_columns(self, df, metric, scaler, clss="stats"):
        if metric.endswith('_'):  # Wildcard matching
            print(f'{utils.colors.MAGENTA}Using wildcard matching for {metric} {utils.colors.ENDC}')
            prefix = clss + "_" + metric[:-1] # Remove the wildcard
            columns_to_compare = [col for col in df.columns if col.startswith(prefix)]
            df[columns_to_compare] = scaler.fit_transform(df[columns_to_compare])
        else:
            print(f'{utils.colors.MAGENTA}Using exact matching for {metric} {utils.colors.ENDC}')
            metric = clss + "_" + metric
            if metric not in df.columns:
                raise ValueError(f"The metric {metric} doesn't exist in the data.")
            df[metric + "_standardized"] = scaler.fit_transform(df[[metric]])
            columns_to_compare = [metric + "_standardized"]
        return columns_to_compare
        
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
        try:
            if(len(similar_files) <= 1):
                return
            # Create a directory for the sound_id
            target_folder = os.path.join(base_path, file_id.split(".")[0])
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            # create a directory for the analysis files
            analysis_folder = os.path.join(target_folder, "analysis")
            if not os.path.exists(analysis_folder):
                os.makedirs(analysis_folder)

            # This assumes that the source directory is one level above the data_path
            source_diectory = os.path.dirname(data_path)

            sound_files = similar_files + [file_id]
            for sound in similar_files:
                if not isinstance(sound, str):
                    break
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
                        print(f'{utils.colors.CYAN} Copying {sound} {utils.colors.ENDC}')
                        shutil.copy2(source_file_path, target_folder)
                    if not os.path.exists(os.path.join(analysis_folder, analysis_file)):
                        shutil.copy2(analysis_file_path, analysis_folder)
                else:
                    debug.log_error(f'File {sound} does not exist')

                await asyncio.sleep(0.005)  # just to mimic some delay
            print(f'{utils.colors.GREEN} Done copying {len(sound_files)} files to {target_folder} {utils.colors.ENDC}\n')
        except Exception as e:
            debug.log_error(f'Error occurred while copying files: {e}')

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
        print(f'{utils.colors.GREEN}Found {len(similar_files)} similar sounds for {primary_file} {utils.colors.ENDC}')
        await self.copy_similar_to_folders(self.base_path, self.data_path, primary_file, similar_files)
        used_files.update(similar_files)
        all_files.difference_update(used_files)
        
    def main(self):
        debug.log_info('Checking directory')
        for sound in os.listdir(self.args.input):
            if utils.check_format(sound):
                sound = os.path.splitext(sound)[0]
                if not os.path.exists(os.path.join(self.data_path, sound + "_analysis.json")):
                    utils.error_logger.error(sound + ' has not been analyzed. Please run wavecraft extract first.')
                    sys.exit(1)
                
        
        data = utils.load_dataset(self.data_path)
        df = pd.json_normalize(data, sep="_")
        all_files = set(df["id"].tolist())
        
        if self.args.n_max == -1:
            self.args.n_max = len(all_files)
            
        if self.args.metric_to_analyze and self.args.metric_range:
            all_within_range = self.find_all_based_on_metric(self.args.identifier, df, self.args.metric_to_analyze, self.args.metric_range, self.args.class_to_analyse)
            debug.log_stat(f'Found {len(all_within_range)} files within the {self.args.metric_to_analyze} range of {self.args.metric_range}')
            asyncio.run(self.copy_similar_to_folders(self.base_path, self.data_path, all_within_range[0], all_within_range))
            return
            
        used_files = set()
        loop = asyncio.get_event_loop()
        
        while all_files and len(used_files) < self.args.n_max:
            loop.run_until_complete(self.process_batch(all_files=all_files, 
                                                used_files=used_files, 
                                                df=df, 
                                                metric=self.args.metric_to_analyze, 
                                                n=self.args.n_similar, 
                                                clss=self.args.class_to_analyse, 
                                                id=self.args.identifier, 
                                                ops=self.ops))

    


    
    
    
    
