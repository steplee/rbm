#include "loader.h"
#include "my_math.h"

typedef uint8_t uchar;


MnistLoader::MnistLoader(std::string im_file, std::string label_file)
 : img_stream(im_file.c_str(), std::fstream::in | std::ifstream::binary)
 , label_stream(label_file.c_str(), std::fstream::in | std::ifstream::binary) {
   num_ims = 0;

   if (not img_stream.good())
     std::cout << "Failed to open img stream." << std::endl;
   if (not label_stream.good())
     std::cout << "Failed to open img stream." << std::endl;

   char nothing;
   for (int i=0;i<4*4; i++)
      img_stream.get(nothing);
   for (int i=0;i<4*2; i++)
      label_stream.get(nothing);
}

MnistLoader::ImageLabelVector MnistLoader::nextBatch(int size) {
  std::vector<std::vector<float>> ims;
  std::vector<int> labels;

  char *xx = new char[28*28];
  float *xxx = new float[28*28];

  for (int b=0; b<size; b++) {

    // Load Image
    img_stream.read(xx, 28*28);
    //img_stream >> xx;
    for (int i=0; i<28*28; i++) {
      xxx[i] = ((float)((uint8_t)xx[i]))/255.0;
      //xxx[i] = min(max(0,((float)((uint8_t)xx[i]))/255.0),1.0);
    }

    std::vector<float> f;
    f.insert(f.begin(), xxx,xxx+28*28);
    ims.push_back(f);

    // Load label
    char _label;
    label_stream.get(_label);
    uchar label = _label;
    labels.push_back((int)label);
  }

  delete[] xx;
  delete[] xxx;

  return {ims, labels};
}


/*
int main() {
  MnistLoader l("/data/mnist/raw/train-images-idx3-ubyte",
                "/data/mnist/raw/train-labels-idx1-ubyte");

  auto b = l.nextBatch(10);

  for (int i=0;i<10;i++) {
  }

  return 0;
}
*/
