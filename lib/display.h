#pragma once

#include <boost/thread.hpp>
#include <boost/serialization/singleton.hpp>

#include <SFML/Graphics.hpp>
#include <SFML/Graphics/Image.hpp>
#include <SFML/Window/Keyboard.hpp>

class Display :
  public boost::serialization::singleton<Display>
{
  private:
    sf::RenderWindow* window;

    sf::Image img;
    sf::Texture texture;
    sf::Sprite sprite;

  public:
    void loop();
    void init();
    void set_pixels_grayscale(const float *pixels);
    boost::mutex mtx;
};
