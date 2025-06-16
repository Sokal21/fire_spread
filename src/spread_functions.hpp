#pragma once

#include <vector>

#include "fires.hpp"
#include "landscape.hpp"

struct SimulationParams {
  float independent_pred;
  float wind_pred;
  float elevation_pred;
  float slope_pred;
  float subalpine_pred;
  float wet_pred;
  float dry_pred;
  float fwi_pred;
  float aspect_pred;
};

Fire simulate_fire(
    const Landscape& landscape, const std::vector<std::pair<size_t, size_t>>& ignition_cells,
    SimulationParams params, float distance, float elevation_mean, float elevation_sd,
    float upper_limit
);

Fire simulate_fire_cuda(
  const Landscape& host_landscape, const std::vector<std::pair<size_t, size_t>>& host_ignition_cells,
  SimulationParams host_params, float distance, float elevation_mean, float elevation_sd,
  float upper_limit
);

struct Coord {
  size_t x;
  size_t y;

  __host__ __device__ bool operator==(const Coord& other) const {
    return x == other.x && y == other.y;
  }

  // Para permitir uso como índice en Matrix, si lo necesitás
  __host__ __device__ operator std::pair<size_t, size_t>() const {
    return { x, y };
  }
};