#ifndef ORTHO_PROPERTY_HPP
#define ORTHO_PROPERTY_HPP

#include <iostream>

using std::ostream;

namespace Ortho {
class Angle {
private:

  static float to_degrees(float radians) { return radians * 180.0f / PI; }

  static float to_radians(float degrees) { return degrees * PI / 180.0f; }

  float value = 0.0f;

public:

  static constexpr float PI = 3.1415926535897932384626433832795;

  explicit Angle() {}

  explicit Angle(const float& degrees) : value(to_radians(degrees)) {}

  float radians() const { return value; }

  float degrees() const { return to_degrees(value); }

  void set_degrees(const float& degrees) { value = to_radians(degrees); }

  void set_radians(const float& radians) { value = radians; }

  friend ostream& operator<<(ostream& os, const Angle& prop) {
    os << prop.value << " rad (" << prop.to_degrees(prop.value) << " deg)";
    return os;
  }
};
} // namespace Ortho
#endif