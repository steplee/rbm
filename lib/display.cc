#include "display.h"

#include <iostream>
using namespace std;

void Display::init() {
  window = new sf::RenderWindow(sf::VideoMode(280,280), "Mnist RBM");
  sf::CircleShape shape(100.f);
  shape.setFillColor(sf::Color::Green);

  float a[28*28];
  for (int i=0; i<28*28; i++) {
    a[i] = (rand()%1000) * .001;
  }
  set_pixels_grayscale(a);
}

void Display::loop()
{

  while (window->isOpen())
  {
    // We do not want to hold lock while sleeping!
    {
      boost::mutex::scoped_lock lock(mtx);

      sf::Event event;
      while (window->pollEvent(event))
      {
        if (event.type == sf::Event::Closed)
          window->close();
        if (event.type == sf::Event::KeyPressed and event.key.code == sf::Keyboard::Escape)
          window->close();
      }

      // Render
      window->clear();
      //window.draw(shape);
      window->draw(sprite);
      window->display();
    }

    // sleep after releasing mutex.
    boost::this_thread::sleep_for(boost::chrono::milliseconds(30)); // 30ms ~ 30fps
  }
}

void Display::set_pixels_grayscale(const float* pixels) {
  boost::mutex::scoped_lock lock(mtx);

  // TODO SIMD
  sf::Uint8 d[28*28*4];
  for (int i=0; i<28*28*4; i+=4) {
    d[i  ] = (sf::Uint8)(pixels[i/4]*255.0);
    d[i+1] = (sf::Uint8)(pixels[i/4]*255.0);
    d[i+2] = (sf::Uint8)(pixels[i/4]*255.0);
    d[i+3] = 255;
  }

  img = sf::Image();
  img.create(28,28, d);
  texture.loadFromImage(img);
  sprite = sf::Sprite(texture);
  sprite.setScale(10,10);
  //cout << "New image MADE!\n";
}
