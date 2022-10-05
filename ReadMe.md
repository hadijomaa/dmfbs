# Hyperparameter Optimzation with Differentiable Meta-features
We provide here the source code for our paper: [Hyperparameter Optimzation with Differentiable Meta-features](https://arxiv.org/abs/2102.03776).
).

## Usage
To meta-train the joint surrogate model, run the metalearna.py file.
```
python metalearn.py 
```

Use the weights with the best validation performance to initialize the surrogate for hyperparameter optimization.

```
python run-aftermetalearn.py
```

## Citing DMFBS
-----------

To cite DMFBS please reference our arXiv paper:


```
@article{jomaa2021hyperparameter,
  title={Hyperparameter optimization with differentiable metafeatures},
  author={Jomaa, Hadi S and Schmidt-Thieme, Lars and Grabocka, Josif},
  journal={arXiv preprint arXiv:2102.03776},
  year={2021}
}
```
