#pragma once

#include <cstddef>
#include <vector>
#include <utility> // for std::pair
#include <string>

#include "landscape.hpp"
#include "matrix.hpp"

struct Fire {
  size_t width;
  size_t height;

  Matrix<bool> burned_layer;

  std::vector<std::pair<size_t, size_t>> burned_ids;

  // Positions in burned_ids where a new step starts
  std::vector<size_t> burned_ids_steps;

  // 🔧 Constructor por defecto seguro (usa tamaño 0)
  Fire() : width(0), height(0), burned_layer(0, 0) {}

  // 🔧 Constructor con tamaño
  Fire(size_t w, size_t h)
      : width(w), height(h), burned_layer(w, h) {}

  // 🔍 Comparación (útil para testing o validación)
  bool operator==(const Fire& other) const {
    return width == other.width &&
           height == other.height &&
           burned_layer == other.burned_layer &&
           burned_ids == other.burned_ids;
  }
};

// 🔎 Función de utilidad para cargar desde archivo
Fire read_fire(size_t width, size_t height, std::string filename);

// 🔎 Estructura y función para estadística del fuego
struct FireStats {
  size_t counts_veg_matorral;
  size_t counts_veg_subalpine;
  size_t counts_veg_wet;
  size_t counts_veg_dry;
};

FireStats get_fire_stats(const Fire& fire, const Landscape& landscape);
