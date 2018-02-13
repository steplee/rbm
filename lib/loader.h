#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>


struct MnistLoader {
  typedef std::pair<std::vector<std::vector<float>>, std::vector<int>> ImageLabelVector;

  std::ifstream img_stream;
  std::ifstream label_stream;
  int32_t num_ims;

  int32_t ims_read = 0;

  ImageLabelVector nextBatch(int batch_size);

  MnistLoader(std::string im_file, std::string label_file);

  char cbuf;
  float fbuf;


};
