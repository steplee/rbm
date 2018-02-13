
#include "display.h"
#include "model.h"

#include <iostream>
using namespace std;

#include <boost/thread.hpp>


void launch_model_thread() {
  cout << " - Model thread launching." << endl;
  Model m(784, 40);
  m.work();
  m.erase();
  cout << " - Model thread exiting." << endl;

  cout << " - Exiting..." << endl;
  exit(0);
}

int main() {

  cout << "Creating display." << endl;
  Display& d = Display::get_mutable_instance();
  d.init();

  cout << "Creating model." << endl;

  boost::thread model_thread{launch_model_thread};
  cout << "Display looping." << endl;
  d.loop();

  model_thread.join();
  cout << "Normal exit." << endl;
  return 0;
}
