#include "spread_functions.hpp"
#include "random_pool.hpp"

#define _USE_MATH_DEFINES
#include <cmath>
#include <random>
#include <vector>
#include <immintrin.h>  // For AVX2 intrinsics

#include "fires.hpp"
#include "landscape.hpp"

float spread_probability(
    const Cell& burning, const Cell& neighbour, SimulationParams params, float angle,
    float distance, float elevation_mean, float inv_elevation_sd, float upper_limit = 1.0f
) {
  float slope_term = sinf(atanf((neighbour.elevation - burning.elevation) / distance));
  float wind_term = cosf(angle - burning.wind_direction);
  float elev_term = (neighbour.elevation - elevation_mean) * inv_elevation_sd;

  float linpred = params.independent_pred;

  if (neighbour.vegetation_type == SUBALPINE) {
    linpred += params.subalpine_pred;
  } else if (neighbour.vegetation_type == WET) {
    linpred += params.wet_pred;
  } else if (neighbour.vegetation_type == DRY) {
    linpred += params.dry_pred;
  }

  linpred += params.fwi_pred * neighbour.fwi;
  linpred += params.aspect_pred * neighbour.aspect;

  linpred += wind_term * params.wind_pred + elev_term * params.elevation_pred +
             slope_term * params.slope_pred;

  float prob = upper_limit / (1.0f + expf(-linpred));

  return prob;
}

Fire simulate_fire(
    const Landscape& landscape, const std::vector<std::pair<size_t, size_t>>& ignition_cells,
    SimulationParams params, float distance, float elevation_mean, float elevation_sd,
    float upper_limit = 1.0f
) {
  // Create a random pool for this simulation
  RandomPool random_pool;

  size_t n_row = landscape.height;
  size_t n_col = landscape.width;

  std::vector<std::pair<size_t, size_t>> burned_ids;

  size_t start = 0;
  size_t end = ignition_cells.size();

  for (size_t i = 0; i < end; i++) {
    burned_ids.push_back(ignition_cells[i]);
  }

  std::vector<size_t> burned_ids_steps;
  burned_ids_steps.push_back(end);

  size_t burning_size = end + 1;

  Matrix<bool> burned_bin = Matrix<bool>(n_col, n_row);

  for (size_t i = 0; i < end; i++) {
    size_t cell_0 = ignition_cells[i].first;
    size_t cell_1 = ignition_cells[i].second;
    burned_bin[{ cell_0, cell_1 }] = 1;
  }

  // Directions for the 8 neighbors declared previous to the loop
  constexpr int moves[2][8] = { { -1, -1, -1, 0, 0, 1, 1, 1 },
                                { -1, 0, 1, -1, 1, -1, 0, 1 } };

  // Angles for the 8 neighbors declared previous to the loop
  constexpr float angles[8] = {
    M_PI * 3.0f / 4.0f, M_PI, M_PI * 5.0f / 4.0f, M_PI / 2.0f, M_PI * 3.0f / 2.0f,
    M_PI / 4.0f,        0.0f, M_PI * 7.0f / 4.0f
  };

  float inv_elevation_sd = 1.0f / elevation_sd;

  while (burning_size > 0) {
    size_t end_forward = end;

    // Loop over burning cells in the cycle
    for (size_t b = start; b < end; b++) {
      size_t burning_cell_0 = burned_ids[b].first;
      size_t burning_cell_1 = burned_ids[b].second;

      const Cell& burning_cell = landscape[{ burning_cell_0, burning_cell_1 }];

      // Prefetch next burning cell if it exists
      if (b + 1 < end) {
        size_t next_cell_0 = burned_ids[b + 1].first;
        size_t next_cell_1 = burned_ids[b + 1].second;
        const Cell& next_cell = landscape[{next_cell_0, next_cell_1}];
        __builtin_prefetch(&next_cell, 0, 3);
      }

      // Loop over neighbors of the focal burning cell
      // Load base coordinates into vectors
      __m256i base_x = _mm256_set1_epi32(burning_cell_0);
      __m256i base_y = _mm256_set1_epi32(burning_cell_1);
      
      // Load moves into vectors directly from memory
      __m256i moves_x = _mm256_loadu_si256((__m256i*)&moves[0]);
      __m256i moves_y = _mm256_loadu_si256((__m256i*)&moves[1]);

      // Calculate neighbor coordinates
      __m256i neighbor_x = _mm256_add_epi32(base_x, moves_x);
      __m256i neighbor_y = _mm256_add_epi32(base_y, moves_y);
      
      // Load bounds for comparison
      __m256i bounds_x = _mm256_set1_epi32(n_col);
      __m256i bounds_y = _mm256_set1_epi32(n_row);
      __m256i minusOne = _mm256_set1_epi32(-1);
      
      // Check if coordinates are in bounds
      __m256i x_in_bounds = _mm256_and_si256(
          _mm256_cmpgt_epi32(neighbor_x, minusOne),
          _mm256_cmpgt_epi32(bounds_x, neighbor_x)
      );
      __m256i y_in_bounds = _mm256_and_si256(
          _mm256_cmpgt_epi32(neighbor_y, minusOne),
          _mm256_cmpgt_epi32(bounds_y, neighbor_y)
      );
      __m256i in_bounds = _mm256_and_si256(x_in_bounds, y_in_bounds);
      
      // Convert to mask for processing
      int mask = _mm256_movemask_epi8(in_bounds);
      // Prefetch valid neighbor cells
      for (int n = 0; n < 8; n++) {
        if (!(mask & (1 << (n * 4)))) continue;  // Skip if out of bounds
        
        // Extract coordinates using array indexing
        int neighbor_cell_0 = ((int*)&neighbor_x)[n];
        int neighbor_cell_1 = ((int*)&neighbor_y)[n];
        
        // Prefetch the neighbor cell
        const Cell& neighbor_cell = landscape[{neighbor_cell_0, neighbor_cell_1}];
        __builtin_prefetch(&neighbor_cell, 0, 3);
      }
      // Process each neighbor that's in bounds
      for (int n = 0; n < 8; n++) {
        if (!(mask & (1 << (n * 4)))) continue;  // Skip if out of bounds
        
        // Extract coordinates using array indexing instead of _mm256_extract_epi32
        int neighbor_cell_0 = ((int*)&neighbor_x)[n];
        int neighbor_cell_1 = ((int*)&neighbor_y)[n];
        
        const Cell& neighbour_cell = landscape[{ neighbor_cell_0, neighbor_cell_1 }];
        
        // Is the cell burnable?
        if (burned_bin[{ neighbor_cell_0, neighbor_cell_1 }] || !neighbour_cell.burnable) {
          continue;
        }
        
        // simulate fire
        float prob = spread_probability(
            burning_cell, neighbour_cell, params, angles[n], distance, elevation_mean,
            inv_elevation_sd, upper_limit
        );
        
        // Burn with probability prob (Bernoulli)
        if (random_pool.get_random() >= prob) {
          continue;
        }
        
        // If burned, store id of recently burned cell and set 1 in burned_bin
        end_forward += 1;
        burned_ids.push_back({ neighbor_cell_0, neighbor_cell_1 });
        burned_bin[{ neighbor_cell_0, neighbor_cell_1 }] = true;
      }
    }

    // update start and end
    start = end;
    end = end_forward;
    burning_size = end - start;

    burned_ids_steps.push_back(end);
  }

  return { n_col, n_row, burned_bin, burned_ids, burned_ids_steps };
}