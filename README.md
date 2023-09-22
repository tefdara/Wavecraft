# Wave-Craft
Audio segmentation, decomposition and more...

Wave-Craft is a Python-based tool for audio manipulation, segmentation and decomposition. It's a unification of a lot of small bash and python tools I made to make my life easier. It is built on top of the [librosa](https://librosa.org) library, which provides a wide range of audio processing and analysis tools. `Wave Craft` has a command-line interface for performing various operations on audio files, including segmenting them into smaller samples based on onsets or other methods, decomposing them using various methods, applying filters, writing metadata, normalising, triming, extracting features etc. 

This is a new project of mine and is a work in progress. I'll be updating and expanding it in the coming months, years, decades, centuries, and millennia.



## Dependencies

`Wave Craft` requires the following modules:

- `Python` 3.6 or higher
- `librosa`
- `soundfile`
- `numpy`
- `scipy`
- `scikit-learn`
- `pandas`
- `pyloudnorm`

Look at the [requirements.txt](requirements.txt) and below for details on how to use the bash script to install the dependencies.

## Usage

First, if you get a permission error, make sure the script is executable:

```shell
chmod +x wac.py
```

To segment an audio file using `wac.py`, run the following command. 

```shell
./wac.py segment input_file.wav output_directory [options]
```

To decompose an audio file using `wac.py`, run the following command:

```shell
./wac.py decompose input_file.wav [options]
```

To perform other operations on an audio file using `wac.py`, run the following command:

```shell
./wac.py operation [options] arg
```

Where `operation` is the operation to perform, `options` are the options for the operation, and `arg` is the path to the audio, metadata, or dataset file. 

Replace `operation_name` with the name of the operation you want to perform. The available operations so far are. All the operations can be done on a single file or a directory of files:

- `segment`: Segment an audio file based on onsets or other methods.
- `extract`: Extract features from an audio file.
- `proxim`: Calculate proximity metrics for a dataset. It can find the most similar sounds in a dataset based on various metrics.
- `decompose`: Decompose an audio file using various methods.
- `filter`: Apply filters to an audio files.
- `norm`: Normalise an audio file.
- `wmetadata`: Write metadata to an audio file.
- `info`: Get information about an audio file.


For more details on the available operations and options, run:

```sh
./wac.py -h
```

## Examples

Here are some more examples of how to use `Wave Craft`:

### Split an audio file into segments

To split an audio file into segments based on a text file, run:

```sh
./wac.py segment -i /path/to/audio/file.wav -it /path/to/text/file.txt -o /path/to/output/directory
```

This will split the audio file into segments based on the text file and save the segments to the output directory.

### Extract features from an audio file

To extract features from an audio file, run:

```sh
./wac.py extract -i /path/to/audio/file.wav -fex mel -fdic True
```

This will extract mel spectrogram features from the audio file and save them to a flattened dictionary.

### Find most similar sounds using proximity metrics

To calculate proximity metrics for a dataset, run:

```sh
./wac.py proxim -i /path/to/dataset -ns 5 -cls stats
```

This will calculate the proximity metrics for the dataset and retrieve the 5 most similar sounds.


Note that some operations may require additional arguments. Use the `--help` option with `wac.py` and the operation name to see the available options for each operation.


Certainly! Here's a simple `README.md` that explains the script and provides an example `requirements.txt`:

---

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
2. **requirements.txt not found**: Ensure that the `requirements.txt` file exists in the root directory of `Wave Craft`.


## License

`Wave Craft` is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for more details.