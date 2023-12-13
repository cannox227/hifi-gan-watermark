# Conda env setup
Conda version used: `conda 23.7.2`

## Steps
1. Create the environment and install the requirements

    ```conda env create --file environment.yml --name <env_name>```

2. Activate the environment
    ```conda activate <env_name>```


## Tips
### Export environment configuration
If you add a new library or modify any library version you can re-export module list by using

    conda env export --no-builds -n <env-name> > environment.yml

### Training
1. Download the dataset in the same project folder
       
        wget http://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
2. Unzip it
    
        tar -xf LJSpeech-1.1.tar.bz2 
3. Train

        python3 train.py --config config_custom.json  

## Other useful commands

### Inference
    python3 inference.py --checkpoint_file LJ_FT_T2_V3/generator_v3

### Training
    python3 train.py --config <config_file.json> --training_epochs=50 --checkpoint_interval=820

### Experiments script
    python3 audio2waveform.py

### Validation
    python3 validation.py --config <config_file.json> --checkpoint_path <model_folder>