#pragma once

#include <vector>
#include <random>

#include "my_math.h"


const int BATCH_SIZE = 20;
const int EPOCHS = 4;
const int SHOW_EVERY = 50;
const int CD_STEPS = 5;


class Model {

  private:
    float *a;
    float *b;
    float *w;
    void init();

    int hid_size;
    int vis_size;

    void train_batch(std::vector<std::vector<float>> imgs);
    void do_contrastive_divergence(int n, float* w_grad,
                                          float* a_grad,
                                          float* b_grad, float* v);

    void sample_h(float* dst, const float* v, bool stochastic_binary=false);
    void sample_v(float* dst, const float* h, bool stochastic_binary=false);

    std::default_random_engine gen;
    std::uniform_real_distribution<float> uni;

  public:
    void work();

    Model(int vsize, int hid_size);

    // Destructor cannot be used for deleting member data in multi-threaded code.
    ~Model();
    // So call this instead (or I could use ref counting, but nvm that)
    void erase();

};
