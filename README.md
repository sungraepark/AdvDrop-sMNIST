# AdvDrop-sMNIST

Tensorflow implementation in the paper "Adversarial Dropout for Reccurent Neural Networks" 

## Requirements
 * Python 2.7
 * Tensorflow 1.4.1
 
## Implementation

```python train.py --seed=101  --method=adv --adv_cost=kl --K=1 --num_adv_change=3 --permuted=False```
