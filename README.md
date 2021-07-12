# Conditional Normalizing Flows

This repository contains a Conditional Invertible Neural Network (impemented with the framework [FrEIA](https://github.com/VLL-HD/FrEIA)) and a Conditional Glow model (implemented with the framework [Full Glow](https://github.com/MoeinSorkhei/glow2)).

Example usage for running the experiment 'Random Mask'

## CINN
```
python train.py --problem randommask --ppar 0.2 --snr 10 --batchsize 256
```
*(Samples are generated every five epochs)*

## C-Glow
Training:
```
python3 main.py --model c_glow_v3 --dataset randommask --n_block 4 --n_flow 2 2 2 2 --do_lu --reg_factor 0.01 --grad_checkpoint
```
Sampling:
```
python3 main.py --model c_glow_v3 --dataset randommask --n_block 4 --n_flow 2 2 2 2 --do_lu --sample
```

Further training sets must be defined in the code in order to avoid error messages. Samples generated from the models can be viewed [here](https://github.com/leonardosalsi/Normalizingflows/tree/main/Samples).
