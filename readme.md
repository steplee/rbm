# Restricted Boltzmann Machine

An RBM in C++. Trained with contrastive divergence.
Just got a new GPU, an NV, so I'll probably implement it in cuda.

A primitive display using SFML (a c++ library)

### Installation / Usage
```
sudo apt-get install libsfml-dev libsfml-system libsfml-graphics
sudo apt-get install libboost libboost-system libboost-thread
cd model...
make && ./app
```

### TODO
  - As an exercise, implement CPU parallelized and cuda parallelized models
  - Implement some quantitative evaluation metrics and plot accuracy
  - Try on different modalities and experiment with semantic hashing

### References
[http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf]()
[https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine#Training_algorithm]()
