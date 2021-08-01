# SED_SSL_DA
A sound event detection system designed for DCASE 2020 task 4, which consists of large amount of weak label and unlabel audio clips.

This work is based on the baseline of DCASE task 4 (https://github.com/turpaultn/dcase20_task4/tree/public_branch/baseline). 

We try to improve SED system in two directions, **semi-supervised learning** and **domain adaptation**.

**Note:** Check if the baseline code works before using our code.

-------------------------------
### Directory structure
SED-via-consistency-training-and-pseudo-labeling/ and DA/ should be placed parallel to baseline/, and we highly recommend renaming SED-via-consistency-training-and-pseudo-labeling/ with SSL/.

The directory structure should be like below:

- dcase20_task4
    - dataset
        - audio
            - train
                - weak
                    - Y--7jZxfzemI_13.000_23.000.wav
                    - ...
                - unlabel_in_domain
                    - Y--5uHDNDKxs_30.000_40.000.wav
                    - ...
                - synthetic20
                    - soundscapes
                        - 00.wav
                        - ...
                    - ...
                - ...
            - validation
                - Y--4gqARaEJE_0.000_10.000.wav
                - ...
            - ...
        - features
            - sr16000_win2048_hop255_mels128_nolog
                - features
                    - train
                        - weak
                            - Y--7jZxfzemI_13.000_23.000.npy
                            - ...
                        - unlabel_in_domain
                            - Y--5uHDNDKxs_30.000_40.000.npy
                            - ...
                        - ...
                    - synthetic20
                        - soundscapes
                            - 00.npy
                            - ...
                        - ...
                    - audio
                        - validation
                            - Y--4gqARaEJE_0.000_10.000.npy
                            - ...
                        - ...
                - ...
        - metadata
            - train
                - weak.tsv
                - unlabel_in_domain.tsv
                - synthetic20
                    - soundscapes.tsv
                    - ...
                - ...
            - validation
                - validation.tsv
                - ...
            - ...
    - baseline
    - SSL
    - DA
    - ...

**Note:** Please refer to README.md in DA/ and SSL/ for the details of **semi-supervised learning** and **domain adaptation**