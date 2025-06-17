#pragma once
#include <cstddef>
#include <utility> // for std::pair
#ifdef __CUDACC__
  #define HOST_DEVICE __host__ __device__
#else
  #define HOST_DEVICE
#endif

#include <vector>

#include "fires.hpp"
#include "landscape.hpp"
#include "matrix.hpp" // For Matrix<size_t>

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


// Declaration for the new top-level GPU orchestrator
Matrix<size_t> burned_amounts_per_cell_on_gpu(
    const Landscape& landscape,
    const std::vector<std::pair<size_t, size_t>>& initial_ignition_cells_template,
    SimulationParams sim_params,
    float distance, float elevation_mean, float elevation_sd, float upper_limit,
    size_t n_replicates
);


struct Coord {
  size_t x;
  size_t y;

  HOST_DEVICE bool operator==(const Coord& other) const {
    return x == other.x && y == other.y;
  }

  // Conversión implícita a std::pair, útil para Matrix<bool>[Coord]
  HOST_DEVICE operator std::pair<size_t, size_t>() const {
    return { x, y };
  }

  // Orden para std::sort si querés usarla en host
  bool operator<(const Coord& other) const {
    return (y < other.y) || (y == other.y && x < other.x);
  }
};