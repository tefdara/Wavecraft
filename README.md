# Wave-Craft
Audio segmentation, decomposition and more

This repository contains code for audio segmentation and decomposition using the [librosa] (https://librosa.org/)library. The `segmenter.py` file contains the main code for segmenting audio files into smaller segments based on onsets and other methods. The `decomposer.py` file contains code for decomposing audio files using various methods. This is a work in progress. I'll be updating and expanidng it in the coming months. 

## Requirements

- Python 3.x
- `librosa` library
- `soundfile` library

## Usage

To segment an audio file, run the following command:

```
python segmenter.py input_file.wav output_directory
```

To decompose an audio file, run the following command:

```
python decomposer.py input_file.wav output_directory
```

## License

This code is licensed under the GNU GENERAL PUBLIC LICENSE. See the `LICENSE` file for more information.
