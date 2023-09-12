```markdown
# Wave-Craft
Audio segmentation, decomposition and more...

Wave-Craft is a Python-based tool for audio manipulation, segmentation and decomposition. It is built on top of the [librosa](https://librosa.org) library, which provides a wide range of audio processing and analysis tools. Wave-Craft provides a command-line interface for performing various operations on audio files, including segmenting them into smaller samples based on onsets or other methods, decomposing them using various methods, applying filters, writing metadata, normalising, triming, extracting features etc. 

Wave-Craft is a new project of mine and is a work in progress. New features and improvements are being added regularly. It is intended for use by audio engineers, musicians, and anyone else who needs to work with large number audio files.


## Requirements

- Python 3.x
- `librosa` library
- `soundfile` library

## Usage

To segment an audio file using `wave_craft.py`, run the following command:

```
python wave_craft.py segment input_file.wav output_directory `[options]`
```

To decompose an audio file using `wave_craft.py`, run the following command:

```
python wave_craft.py decompose input_file.wav `[options]`
```

To perform other operations on an audio file using `wave_craft.py`, run the following command:

```
python wave_craft.py operation input_file.wav 
```

Replace `operation_name` with the name of the operation you want to perform. The available operations so far are:

- `segment`: Segment an audio file based on onsets or other methods.
- `decompose`: Decompose an audio file using various methods.
- `hpf`: Apply a high-pass filter to an audio file.
- `wmetadata`: Write metadata to an audio file.
- `info`: Get information about an audio file.

For example, to apply a high-pass filter to an audio file, run the following command:

```
python wave_craft.py hpf input_file.wav -o output_directory
```

This will apply a high-pass filter to the audio file and save the result in the `output_directory`.

Note that some operations may require additional arguments. Use the `--help` option with `wave_craft.py` and the operation name to see the available options for each operation.

This is a work in progress. I'll be updating and expanding it in the coming months. 
```


