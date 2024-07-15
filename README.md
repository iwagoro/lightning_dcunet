# lightning_dcunet

Speech enchancement without labels.
Coded with pytorch lightning.

# Setup

You need to install following libraries

```
numpy
pandas
matplotlib
seaborn
scipy
pesq
pytorch-lightning
lightning
tensorboard
torchinfo
thefuck
librosa
rich
black
pydub
python-dotenv

```

# Start up

## args

| arg            | option                             | default | required |
| -------------- | ---------------------------------- | ------- | -------- |
| `--mode`       | `train , predict`                  | None    | yes      |
| `--dataset`    | `white , urban0 , urban1 , urban2` | None    | yes      |
| `--loss`       | `nct , nb2nb`                      | None    | yes      |
| `--model`      | `dcunet , dcunet-rtstm`            | None    | yes      |
| `--devices`    | `0 ~ 8`                            | 0       | no       |
| `--checkpoint` | `your checkpoint path`             | None    | no       |

## exapmple

```bash
python3 trainlpy --mode train --dataset white --loss nb2nb --model dcunet --devices 0 1 2 3 4 5 6 7

```
