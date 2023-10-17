# Conda env setup
Conda version used: `conda 23.7.2`

## Steps
1. Create the environment and install the requirements

    ```conda env create --file environment.yml --name <env_name>```

2. Activate the environment
    ```conda activate <env_name>```

## Test commands
    python3 inference.py --checkpoint_file LJ_FT_T2_V3/generator_v3