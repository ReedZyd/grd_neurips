# Generative Return Decomposition (GRD)

This is a TensorFlow implementation for our NeurIPS 2023 underview paper: 

GRD: A Generative Approach for Interpretable Reward Redistribution in Reinforcement Learning

## Requirements
The requirements are listed as follows:
1. Python 3.7
2. gym == 0.18.3
3. TensorFlow == 2.11.0
4. BeautifulTable == 0.8.0
5. opencv-python == 4.5.3.56
6. wandb==0.13.9
7. mujoco_py (mujoco 210, conda install -c conda-forge glew, pip install patchelf)
8. imageio==2.3.0

## Training & Evaluation
To reproduce the results, please run,
```shell
python train.py \
    --env ${env} --alg causal --policy-learning-with-causal  \
    --zr-sparsity-coef ${coef1} --ar-sparsity-coef ${coef2}--zz-sparsity-coef ${coef3} --zz-sparsity-coef-aux ${coef4} --az-sparsity-coef ${coef5}
```

For different environments, please use hyper-parameters as follows.

| env                |  coef1 |  coef2 |  coef3 |  coef4 |  coef5 |
| ------------------ | ------ | ------ | ------ | ------ | ------ | 
| Ant-v2             |   1e-5 |   1e-9 |   1e-7 |   1e-8 |   1e-8 |
| HalfCheetah-v2     |   1e-5 |   1e-5 |   1e-5 |   1e-6 |   1e-5 |
| Walker2d-v2        |   1e-5 |   1e-5 |   1e-6 |   1e-6 |   1e-7 |
| Humanoid-v2        |   1e-5 |   1e-8 |   1e-5 |   1e-7 |   1e-8 |
| Reacher-v2         |   5e-7 |   1e-8 |   1e-8 |   1e-8 |   1e-8 |
| Swimmer-v2         |   1e-7 |   1e-9 |   1e-9 |   0    |   1e-9 |
| Hopper-v2          |   1e-6 |   1e-6 |   1e-6 |   1e-7 |   1e-6 |
| HumanoidStandup-v2 |   1e-5 |   1e-4 |   1e-6 |   1e-7 |   1e-7 |

