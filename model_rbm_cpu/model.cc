#include "model.h"
#include "display.h"
#include "loader.h"

#include <iostream>
using namespace std;

#include <boost/thread.hpp>

#define LOG_LVL 10


/*
 * http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
 * https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine#Training_algorithm
 */

Model::Model(int vsize, int hsize)
  : hid_size(hsize)
  , vis_size(vsize)
  , uni(0.0f,1.0f)
{
  init();
}

Model::~Model() {
}

void Model::erase() {
  delete[] a;
  delete[] b;
  delete[] w;
}

void Model::init() {
  // From 8.1 of Hinton's doc
  a = new float[vis_size]();
  b = new float[hid_size]();

  std::normal_distribution<float> dist(0.0f, .01f);

  for (int i=0; i<vis_size; i++)
    //a[i] = log(.4f/.6f); // Hinton recommends building data stats, this is an approx.
    a[i] = .000001f; 


  w = new float[vis_size*hid_size];

  for (int i=0; i<vis_size*hid_size; i++)
    w[i] = dist(gen);
}


void Model::work() {
  for (int epoch = 0; epoch < EPOCHS; epoch++) {
    cout << " -- Beginning epoch " << epoch << ".\n";

    // Create loader
    MnistLoader loader("/data/mnist/raw/train-images-idx3-ubyte",
                       "/data/mnist/raw/train-labels-idx1-ubyte");

    //boost::this_thread::sleep_for(boost::chrono::milliseconds(900));

    Display& disp = Display::get_mutable_instance();

    for (int batch_id = 0; batch_id < 60000/BATCH_SIZE; batch_id++) {
      // <vec<img> , vec<label>>
      auto xs = loader.nextBatch(10);

      train_batch(xs.first);


      // Display a generated sample
      if (batch_id % SHOW_EVERY == 0) {
        float *h_ = new float[hid_size];
        float *v_ = new float[vis_size];

        // Randomly sample h.
        //for (int i=0; i<hid_size; i++) h_[i] = uni(gen);
        //sample_v(v_, h_, false);

        // Full n-Step Gibbs sampling.
        memcpy(v_, xs.first[0].data(), sizeof(float)*vis_size);
        for (int i=0; i<CD_STEPS; i++)
          sample_h(h_, xs.first[0].data(), false),
          sample_v(v_, h_, false);

        //disp.set_pixels_grayscale(xs.first[0].data());
        disp.set_pixels_grayscale(v_);
        delete[] h_;
        delete[] v_;
      }


    }
  }
}

void Model::train_batch(std::vector<std::vector<float>> imgs) {
  // there is no point in mini-batching until I add parallelism obviously
  // but I might as well code it up with batching for easy conversion

  // Cool cpp trick I just learned adding `()` here will behave like calloc for scalar types
  float *w_grad = new float[vis_size*hid_size]();
  float *a_grad = new float[vis_size]();
  float *b_grad = new float[hid_size]();

  for (int im=1; im<imgs.size(); im++) {
    do_contrastive_divergence(CD_STEPS, w_grad, a_grad, b_grad, imgs[im].data());
  }

  float norm_w = 0, norm_a = 0, norm_b = 0;
  for (int i=0; i<vis_size*hid_size; i++)
    w[i] += w_grad[i],
    norm_w += w_grad[i]*w_grad[i];
  for (int i=0; i<vis_size; i++)
    a[i] += a_grad[i],
    norm_a += a_grad[i]*a_grad[i];
  for (int i=0; i<hid_size; i++)
    b[i] += b_grad[i],
    norm_b += b_grad[i]*b_grad[i];

  norm_w = sqrt(norm_w); norm_a = sqrt(norm_a); norm_b = sqrt(norm_b);
  if (LOG_LVL >= 1)
    printf("Updating with norms: w:%f\ta:%f\tb:%f\n", norm_w,norm_a,norm_b);

  delete[] w_grad;
  delete[] a_grad;
  delete[] b_grad;

}

void Model::sample_h(float* dst, const float* v, bool stochastic_binary) {
  for (int j=0; j<hid_size; j++) {
    float sum_vw = 0;
    for (int i=0; i<vis_size; i++)
      sum_vw += v[i]*w[i*hid_size+j];


    if (stochastic_binary)
      dst[j] = uni(gen) < sigmoid(b[j] + sum_vw) ? 1.0 : 0.0;
    else
      dst[j] = sigmoid(b[j] + sum_vw);
  }
}

void Model::sample_v(float* dst, const float* h, bool stochastic_binary) {
  for (int i=0; i<vis_size; i++) {
    float sum_hw = 0;
    for (int j=0; j<hid_size; j++)
      sum_hw += h[j]*w[i*hid_size+j];

    if (stochastic_binary)
      dst[i] = uni(gen) < sigmoid(a[i] + sum_hw) ? 1.0 : 0.0;
    else
      dst[i] = sigmoid(a[i] + sum_hw);
  }
}



void Model::do_contrastive_divergence(int n, float* w_grad,
                                             float* a_grad,
                                             float* b_grad, float* v_data) {
  float lr = .01;
  float lr_bias = .001;

  // sampling process
  // v_data -> h_first -> [v_i -> h_i] ...


  // 1. Sample p(h|v)
  float *h_first = new float[hid_size];
  // NOTE: Apparently collapsing prob to a *binary value* is very important here
  // It puts a bottleneck on the learning process (think of an AE) that helps learning.
  // See Hinton doc section 3.4
  sample_h(h_first, v_data, true);

  // 2. Sample v' and h'
  float *v_i = new float[vis_size];
  float *h_i = new float[hid_size];
  memcpy(h_i, h_first, sizeof(float)*hid_size); // h_0 = h_first
  // More Gibbs sampling steps -> better updates
  for (int step=0; step<n; step++) {
    sample_v(v_i, h_i);
    sample_h(h_i, v_i);
  }

  // 3. Positive Phase term:  outer product of (v,h)
  float *vh_t = new float[vis_size*hid_size];
  outer_product(vh_t,   v_data,h_first,   vis_size,hid_size);

  // 4. Negative Phase term: outer of (v',h')
  float *vphp_t = new float[vis_size*hid_size];
  outer_product(vphp_t,   v_i,h_i,    vis_size,hid_size);

  // 5. Step gradient
  for (int i=0; i<vis_size*hid_size; i++)
    w_grad[i] += (vh_t[i] - vphp_t[i]) * lr;
  for (int i=0; i<vis_size; i++) {
    //cout << ((v_data[i] - v_i[i]) * lr_bias) << endl;
    a_grad[i] += (v_data[i] - v_i[i]) * lr_bias;
  }
  for (int i=0; i<hid_size; i++)
    b_grad[i] += (h_first[i] - h_i[i]) * lr_bias;

  delete[] h_first;
  delete[] v_i;
  delete[] h_i;
  delete[] vh_t;
  delete[] vphp_t;

}
