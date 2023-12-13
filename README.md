# Wavecraft

Wavecraft is a Python-based tool for audio manipulation, segmentation and batch processing. It started as a unification of a lot of small bash and python tools I have made over time to make my life easier. However, it is slowly growing into a more comprehensive tool for audio processing. Wavecraft has a command-line interface for performing various operations on audio files. In its simplest form, it requires an operation name and an input file. The input can be a directory of files for batch processing. It can also take a text file as input for some operations. See the [usage](#usage) section for more details.

Wavecraft is a work in progress. I'll be updating and expanding it in the coming months, years and... decades? Pull the latest for updates and bug fixes. I'll be adding more operations and features if time permits. If you have any suggestions or feedback, please let me know.



## Dependencies

`Wavecraft` requires the following modules:

- `Python` 3.6 or higher
- `librosa`
- `soundfile`
- `numpy`
- `scipy`
- `scikit-learn`
- `pandas`
- `pyloudnorm`
- `sounddevice`

Look at the [requirements.txt](requirements.txt) and [dependency installer](#python-dependency-installation-script) for details on how to use the bash script to install the dependencies. It will hopefully become a pip package in the future, but for now, you can use the bash script to install the dependencies.

## Usage

First, if you get a permission error, make sure the script is executable:

```shell
chmod +x wac.py
```

To perform operations on an audio file using `wac.py`, run the following command:

```shell
./wac.py operation [options] arg
```

Where `operation` is the operation to perform, `options` are the options for the operation, and `arg` is the path to the audio, metadata, or dataset file. 

Replace `operation_name` with the name of the operation you want to perform. The available operations so far are. All the operations can be done on a single file or batch process a directory of files:

- `segment`: Segment an audio file based on onsets or other methods.
- `extract`: Extract features from an audio file.
- `proxim`: Calculate proximity metrics for a dataset. It can find the most similar sounds in a dataset based on various metrics.
- `decompose`: Decompose an audio file using various methods.
- `beat`: Extract beats from an audio file.
- `onset`: Extract onsets from an audio file.
- `filter`: Apply filters to an audio files.
- `fade`: Apply fades to an audio file.
- `norm`: Normalise an audio file.
- `wmetadata`: Write metadata to an audio file.
- `trim`: Trim an audio file.
- `split`: Split an audio file into smaller files.
- `pan`: Pan an audio file or convert a multichannel file to mono.



For a detailed list of operations and their options, run:

```sh
./wac.py -h
```

## Examples

Here are some examples of how to use `Wavecraft`:

### Segment an audio file based on onsets / beats using different onset envelopes

```shell
./wac.py segment input_file.wav [options]
```
```shell
./wac.py segment input_file.wav -t 0.2 -oe mel -ts -70 -ff 50
```

This will segment the audio file into smaller samples based on onsets using a mel spectogram as the onset envelope. It will use a peak threshold of 0.2 and trims silence from both ends of the file if its below -70db and apply a high-pass filter with a cutoff frequency of 50Hz.

### Split an audio file into segments using a text file

To split an audio file into segments based on a text file, run:

```sh
./wac.py segment /path/to/audio/file.wav t /path/to/text/file.txt
```

This will split the audio file into segments based on the text file and save the segments to the output directory.

### Extract features from an audio file

To extract features from an audio file, run:

```sh
./wac.py extract /path/to/audio/file.wav -fdic True
```

This will extract all the features from the audio file and save them to a flattened dictionary in a JSON file.

### Find most similar sounds using proximity metrics

To calculate proximity metrics for a dataset, run:

```sh
./wac.py proxim /path/to/dataset -ns 5 -cls stats
```

This will calculate the proximity metrics for the dataset and retrieve the 5 most similar sounds.


### Decomposing an audio file

```shell
./wac.py decompose input_file.wav [options]
```


Note that some operations may require additional arguments. Use the `--help` option with `wac.py` and the operation name to see the available options for each operation.

# Python Dependency Installation Script

This repository contains a Bash script to automatically install Python dependencies from a `requirements.txt` file.

## Requirements

1. Python
2. Pip

## Usage

1. Ensure you have the `requirements.txt` file in your project directory with the necessary Python packages listed. It should be in the repo root directory.

It should look like this:

```
librosa>=0.10.1
numpy>=1.24.4
```

2. Make the script executable:
    ```shell
    chmod +x install_deps.sh
    ```
4. Run the script:
    ```shell
    ./install_deps.sh
    ```

## Troubleshooting

1. **pip not found**: Make sure `pip` is installed. You might need to install or update it.
2. **requirements.txt not found**: Ensure that the `requirements.txt` file exists in the root directory of `Wavecraft`.


## License

`Wavecraft` is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for more details.