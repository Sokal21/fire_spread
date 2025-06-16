// filepath: /home/tomas/Desktop/fire_spread/src/spread_functions.cpp
#include "spread_functions.hpp"
#include "random_pool.hpp"

#define _USE_MATH_DEFINES
#include <cmath>
#include <random>
#include <vector>
#include <immintrin.h>  // For AVX2 intrinsics
#include <array>        // For std::array
#include <cstdint>      // For uint8_t
#include <algorithm>    // For std::sort

#include "fires.hpp"
#include "landscape.hpp"
#include "omp.h"


// Scalar version (can be kept for reference or fallback)
float spread_probability_scalar(
    const Cell& burning, const Cell& neighbour, SimulationParams params, float angle,
    float distance, float elevation_mean, float inv_elevation_sd, float upper_limit = 1.0f
) {
  // ... (original scalar implementation) ...
  float slope_term = sinf(atanf((neighbour.elevation - burning.elevation) / distance));
  float wind_term = cosf(angle - burning.wind_direction);
  float elev_term = (neighbour.elevation - elevation_mean) * inv_elevation_sd;

  float linpred = params.independent_pred;

  // Simplified vegetation check
  if (neighbour.vegetation_type == SUBALPINE) {
    linpred += params.subalpine_pred;
  } else if (neighbour.vegetation_type == WET) {
    linpred += params.wet_pred;
  } else if (neighbour.vegetation_type == DRY) {
    linpred += params.dry_pred;
  } // Assuming MATORRAL adds 0 or is baseline

  linpred += params.fwi_pred * neighbour.fwi;
  linpred += params.aspect_pred * neighbour.aspect;

  linpred += wind_term * params.wind_pred + elev_term * params.elevation_pred +
             slope_term * params.slope_pred;

  // Sigmoid calculation
  float prob = upper_limit / (1.0f + expf(-linpred));

  return prob;
}

// Helper function for vectorized expf (requires AVX support and potentially libmvec or similar)
// This is a placeholder; actual implementation depends on compiler/libraries.
// Compilers with -ffast-math might auto-vectorize expf in loops.
// For explicit control, you might use libraries like SLEEF or Intel SVML.
// A simple manual approach often involves polynomial approximation.
// Assuming compiler vectorizes expf or provides an intrinsic like _mm256_exp_ps
#ifndef _mm256_exp_ps
#define _mm256_exp_ps(a) _mm256_set_ps(expf(((float*)&a)[7]), expf(((float*)&a)[6]), expf(((float*)&a)[5]), expf(((float*)&a)[4]), expf(((float*)&a)[3]), expf(((float*)&a)[2]), expf(((float*)&a)[1]), expf(((float*)&a)[0]))
#endif
 #ifndef _mm256_cos_ps // Placeholder
#define _mm256_cos_ps(a) _mm256_set_ps(cosf(((float*)&a)[7]), cosf(((float*)&a)[6]), cosf(((float*)&a)[5]), cosf(((float*)&a)[4]), cosf(((float*)&a)[3]), cosf(((float*)&a)[2]), cosf(((float*)&a)[1]), cosf(((float*)&a)[0]))
#endif

Fire simulate_fire(
    const Landscape& landscape, const std::vector<std::pair<size_t, size_t>>& ignition_cells,
    SimulationParams params, float distance, float elevation_mean, float elevation_sd,
    float upper_limit = 1.0f
) {
  // RandomPool random_pool; // Will be created per thread

  size_t n_row = landscape.height;
  size_t n_col = landscape.width;

  std::vector<std::pair<size_t, size_t>> burned_ids;
  burned_ids.reserve(n_row * n_col);

  for (const auto& cell : ignition_cells) {
    burned_ids.push_back(cell);
  }

  std::vector<size_t> burned_ids_steps;
  if (!ignition_cells.empty()) {
      burned_ids_steps.push_back(burned_ids.size());
  } else {
      burned_ids_steps.push_back(0);
  }


  Matrix<bool> burned_bin(n_col, n_row); // Using Matrix<bool> specialization
  // Initialize burned_bin to false (0)
  // The Matrix<bool> constructor with std::vector<bool> elems(width*height) initializes to false.
  // If using bitmask version, ensure it initializes to 0.
  // For std::vector<bool>, direct fill might be needed if constructor doesn't guarantee false.
  // Assuming Matrix<bool> default constructor or its `elems` init to false.
  // If not, add: std::fill(burned_bin.elems.begin(), burned_bin.elems.end(), false);
  // Or if using the bitmask version: burned_bin.fill(false);


  for (const auto& cell : ignition_cells) {
    if (cell.first < n_col && cell.second < n_row) {
         burned_bin[cell] = true;
    }
  }

  size_t start = 0;
  // 'end' will now represent the count of cells processed in the current step's burned_ids list.
  // The actual end of burned_ids vector will grow.
  size_t current_step_burning_count = ignition_cells.size();


  constexpr int moves[2][8] = { { -1, -1, -1, 0, 0, 1, 1, 1 },
                                { -1, 0, 1, -1, 1, -1, 0, 1 } };
  alignas(32) constexpr float angles[8] = {
    M_PI * 3.0f / 4.0f, M_PI, M_PI * 5.0f / 4.0f, M_PI / 2.0f, M_PI * 3.0f / 2.0f,
    M_PI / 4.0f,        0.0f, M_PI * 7.0f / 4.0f
  };

  float inv_elevation_sd = (elevation_sd != 0.0f) ? 1.0f / elevation_sd : 0.0f;

  // --- AVX Constants (remains the same) ---
  const __m256 v_inv_dist = (distance != 0.0f) ? _mm256_set1_ps(1.0f / distance) : _mm256_setzero_ps();
  const __m256 v_elev_mean = _mm256_set1_ps(elevation_mean);
  const __m256 v_inv_elev_sd = _mm256_set1_ps(inv_elevation_sd);
  const __m256 v_upper_limit = _mm256_set1_ps(upper_limit);
  const __m256 v_one = _mm256_set1_ps(1.0f);
  const __m256 v_zero = _mm256_setzero_ps();
  const __m256i v_minus_one_i = _mm256_set1_epi32(-1);
  const __m256i v_bounds_x = _mm256_set1_epi32(n_col);
  const __m256i v_bounds_y = _mm256_set1_epi32(n_row);
  const __m256 v_angles = _mm256_load_ps(angles);

  const __m256 v_param_indep = _mm256_set1_ps(params.independent_pred);
  const __m256 v_param_suba = _mm256_set1_ps(params.subalpine_pred);
  const __m256 v_param_wet = _mm256_set1_ps(params.wet_pred);
  const __m256 v_param_dry = _mm256_set1_ps(params.dry_pred);
  const __m256 v_param_fwi = _mm256_set1_ps(params.fwi_pred);
  const __m256 v_param_aspect = _mm256_set1_ps(params.aspect_pred);
  const __m256 v_param_wind = _mm256_set1_ps(params.wind_pred);
  const __m256 v_param_elev = _mm256_set1_ps(params.elevation_pred);
  const __m256 v_param_slope = _mm256_set1_ps(params.slope_pred);

  // Temporary storage arrays are now declared inside the parallel region or made thread-private if needed.
  // However, they are small and used per-iteration of the 'b' loop, so they can be stack-allocated within the loop
  // or declared inside the #pragma omp parallel block to be private by default for stack variables.

  while (current_step_burning_count > 0) {
    size_t end_of_current_step_processing = start + current_step_burning_count;
    std::vector<std::pair<size_t, size_t>> all_newly_ignited_coords_for_this_step;
    // Pre-allocate based on a heuristic, e.g., max 8 * number of burning cells
    all_newly_ignited_coords_for_this_step.reserve(current_step_burning_count * 8);


    #pragma omp parallel
    {
      // Each thread gets its own RandomPool instance
      RandomPool thread_local_random_pool;
      // Each thread collects its newly found coordinates here
      std::vector<std::pair<size_t, size_t>> thread_private_new_coords;
      thread_private_new_coords.reserve(8 * (current_step_burning_count / omp_get_num_threads() + 1)); // Heuristic

      // Declare these temporary arrays here to make them private to each thread's stack
      alignas(32) std::array<float, 8> neigh_elev;
      alignas(32) std::array<float, 8> neigh_fwi;
      alignas(32) std::array<float, 8> neigh_aspect;
      alignas(32) std::array<uint8_t, 8> neigh_veg_type;
      alignas(32) std::array<uint8_t, 8> neigh_burnable;
      alignas(32) std::array<uint8_t, 8> neigh_already_burned; // This reads from shared burned_bin
      alignas(32) std::array<int32_t, 8> neigh_x_coords;
      alignas(32) std::array<int32_t, 8> neigh_y_coords;

      #pragma omp for schedule(dynamic)
      for (size_t b = start; b < end_of_current_step_processing; b++) {
        size_t burning_cell_0 = burned_ids[b].first; // Reading from shared burned_ids
        size_t burning_cell_1 = burned_ids[b].second;

        // landscape is shared const, burned_bin is shared (read-only in this parallel part)
        const Cell& burning_cell = landscape[{ burning_cell_0, burning_cell_1 }];

        // Prefetch can remain, ensure it's safe if multiple threads prefetch same/nearby data
        if (b + 1 < end_of_current_step_processing) {
           const Cell& next_cell = landscape[{burned_ids[b + 1].first, burned_ids[b + 1].second}];
           _mm_prefetch((const char*)(&next_cell), _MM_HINT_T0);
        }

        // --- Vectorized Neighbor Calculation (largely the same) ---
        __m256i base_x = _mm256_set1_epi32(burning_cell_0);
        __m256i base_y = _mm256_set1_epi32(burning_cell_1);
        __m256i moves_x_vec = _mm256_loadu_si256((__m256i const*)moves[0]); // Renamed to avoid conflict
        __m256i moves_y_vec = _mm256_loadu_si256((__m256i const*)moves[1]); // Renamed
        __m256i neighbor_x = _mm256_add_epi32(base_x, moves_x_vec);
        __m256i neighbor_y = _mm256_add_epi32(base_y, moves_y_vec);
        _mm256_store_si256((__m256i*)neigh_x_coords.data(), neighbor_x);
        _mm256_store_si256((__m256i*)neigh_y_coords.data(), neighbor_y);

        // --- Bounds Check (same) ---
        __m256i x_gt_m1 = _mm256_cmpgt_epi32(neighbor_x, v_minus_one_i);
        __m256i x_lt_max = _mm256_cmpgt_epi32(v_bounds_x, neighbor_x);
        __m256i y_gt_m1 = _mm256_cmpgt_epi32(neighbor_y, v_minus_one_i);
        __m256i y_lt_max = _mm256_cmpgt_epi32(v_bounds_y, neighbor_y);
        __m256i in_bounds_x = _mm256_and_si256(x_gt_m1, x_lt_max);
        __m256i in_bounds_y = _mm256_and_si256(y_gt_m1, y_lt_max);
        __m256i v_in_bounds_mask = _mm256_and_si256(in_bounds_x, in_bounds_y);

        // --- Gather Neighbor Data (largely same, reads from shared landscape and burned_bin) ---
        std::fill(neigh_already_burned.begin(), neigh_already_burned.end(), 1);
        std::fill(neigh_burnable.begin(), neigh_burnable.end(), 0);
        int in_bounds_bitmask = _mm256_movemask_epi8(v_in_bounds_mask);
        for (int n = 0; n < 8; ++n) {
            if (in_bounds_bitmask & (1 << (n * 4))) {
                int nx = neigh_x_coords[n];
                int ny = neigh_y_coords[n];
                const Cell& neighbour_cell = landscape[{ (size_t)nx, (size_t)ny }];
                neigh_elev[n] = neighbour_cell.elevation;
                neigh_fwi[n] = neighbour_cell.fwi;
                neigh_aspect[n] = neighbour_cell.aspect;
                neigh_veg_type[n] = static_cast<uint8_t>(neighbour_cell.vegetation_type);
                neigh_burnable[n] = neighbour_cell.burnable ? 1 : 0;
                neigh_already_burned[n] = burned_bin[{ (size_t)nx, (size_t)ny }]; // Read from shared burned_bin
            } else {
                neigh_elev[n] = 0.0f; neigh_fwi[n] = 0.0f; neigh_aspect[n] = 0.0f;
                neigh_veg_type[n] = 0;
            }
        }

        // --- Load Gathered Data into AVX Registers (same) ---
        __m256 v_neigh_elev = _mm256_load_ps(neigh_elev.data());
        __m256 v_neigh_fwi = _mm256_load_ps(neigh_fwi.data());
        __m256 v_neigh_aspect = _mm256_load_ps(neigh_aspect.data());
        __m128i v_neigh_burnable_8bit = _mm_loadu_si64(neigh_burnable.data());
        __m128i v_neigh_already_burned_8bit = _mm_loadu_si64(neigh_already_burned.data());
        __m256i v_neigh_burnable_ext = _mm256_cvtepu8_epi32(v_neigh_burnable_8bit); // Renamed
        __m256i v_neigh_already_burned_ext = _mm256_cvtepu8_epi32(v_neigh_already_burned_8bit); // Renamed
        __m256i v_is_burnable_mask = _mm256_cmpeq_epi32(v_neigh_burnable_ext, _mm256_set1_epi32(1));
        __m256i v_not_burned_mask = _mm256_cmpeq_epi32(v_neigh_already_burned_ext, _mm256_setzero_si256());
        __m256i v_valid_neighbor_mask_i = _mm256_and_si256(v_in_bounds_mask, v_is_burnable_mask);
        v_valid_neighbor_mask_i = _mm256_and_si256(v_valid_neighbor_mask_i, v_not_burned_mask);
        __m256 v_valid_neighbor_mask = _mm256_castsi256_ps(v_valid_neighbor_mask_i);

        // --- Vectorized Probability Calculation (largely same, use thread_local_random_pool) ---
        __m256 v_burn_elev = _mm256_set1_ps(burning_cell.elevation);
        __m256 v_burn_wind_dir = _mm256_set1_ps(burning_cell.wind_direction);
        __m256 v_elev_diff = _mm256_sub_ps(v_neigh_elev, v_burn_elev);
        __m256 v_slope_arg = _mm256_mul_ps(v_elev_diff, v_inv_dist);
        __m256 v_slope_arg_sq = _mm256_mul_ps(v_slope_arg, v_slope_arg);
        __m256 v_one_plus_slope_arg_sq = _mm256_add_ps(v_one, v_slope_arg_sq);
        __m256 v_sqrt_term = _mm256_sqrt_ps(v_one_plus_slope_arg_sq);
        __m256 v_slope_term = _mm256_div_ps(v_slope_arg, v_sqrt_term);
        __m256 v_wind_arg = _mm256_sub_ps(v_angles, v_burn_wind_dir);
        __m256 v_wind_term = _mm256_cos_ps(v_wind_arg);
        __m256 v_elev_term = _mm256_sub_ps(v_neigh_elev, v_elev_mean);
        v_elev_term = _mm256_mul_ps(v_elev_term, v_inv_elev_sd);
        __m256 v_linpred = v_param_indep;
        __m128i v_neigh_veg_type_8bit = _mm_loadu_si64(neigh_veg_type.data());
        __m256i v_veg_type = _mm256_cvtepu8_epi32(v_neigh_veg_type_8bit);
        __m256i v_is_suba_mask_i = _mm256_cmpeq_epi32(v_veg_type, _mm256_set1_epi32(SUBALPINE));
        __m256 v_is_suba_mask = _mm256_castsi256_ps(v_is_suba_mask_i);
        v_linpred = _mm256_add_ps(v_linpred, _mm256_and_ps(v_is_suba_mask, v_param_suba));
        __m256i v_is_wet_mask_i = _mm256_cmpeq_epi32(v_veg_type, _mm256_set1_epi32(WET));
        __m256 v_is_wet_mask = _mm256_castsi256_ps(v_is_wet_mask_i);
        v_linpred = _mm256_add_ps(v_linpred, _mm256_and_ps(v_is_wet_mask, v_param_wet));
        __m256i v_is_dry_mask_i = _mm256_cmpeq_epi32(v_veg_type, _mm256_set1_epi32(DRY));
        __m256 v_is_dry_mask = _mm256_castsi256_ps(v_is_dry_mask_i);
        v_linpred = _mm256_add_ps(v_linpred, _mm256_and_ps(v_is_dry_mask, v_param_dry));
        v_linpred = _mm256_fmadd_ps(v_param_fwi, v_neigh_fwi, v_linpred);
        v_linpred = _mm256_fmadd_ps(v_param_aspect, v_neigh_aspect, v_linpred);
        v_linpred = _mm256_fmadd_ps(v_param_wind, v_wind_term, v_linpred);
        v_linpred = _mm256_fmadd_ps(v_param_elev, v_elev_term, v_linpred);
        v_linpred = _mm256_fmadd_ps(v_param_slope, v_slope_term, v_linpred);
        __m256 v_neg_linpred = _mm256_sub_ps(v_zero, v_linpred);
        __m256 v_exp_term = _mm256_exp_ps(v_neg_linpred); // Corrected to natural exp
        __m256 v_denom = _mm256_add_ps(v_one, v_exp_term);
        __m256 v_prob = _mm256_div_ps(v_upper_limit, v_denom);

        // --- Vectorized Random Check (use thread_local_random_pool) ---
        __m256 v_random = thread_local_random_pool.get_random_avx();
        __m256 v_burn_mask = _mm256_cmp_ps(v_random, v_prob, _CMP_LT_OQ);
        __m256 v_final_burn_mask = _mm256_and_ps(v_valid_neighbor_mask, v_burn_mask);
        int final_bitmask = _mm256_movemask_ps(v_final_burn_mask);

        // --- Collect new cells for this thread ---
        if (final_bitmask != 0) {
            for (int n = 0; n < 8; ++n) {
                if ((final_bitmask >> n) & 1) {
                    thread_private_new_coords.push_back({(size_t)neigh_x_coords[n], (size_t)neigh_y_coords[n]});
                }
            }
        }
      } // End of omp for loop

      // --- Merge thread-private results into a shared collection ---
      #pragma omp critical
      {
        all_newly_ignited_coords_for_this_step.insert(
            all_newly_ignited_coords_for_this_step.end(),
            std::make_move_iterator(thread_private_new_coords.begin()),
            std::make_move_iterator(thread_private_new_coords.end())
        );
      }
    } // End of omp parallel region

    // --- Post-parallel processing (single thread) ---
    // Remove duplicates that might have been added if multiple burning cells ignite the same neighbor
    std::sort(all_newly_ignited_coords_for_this_step.begin(), all_newly_ignited_coords_for_this_step.end());
    all_newly_ignited_coords_for_this_step.erase(
        std::unique(all_newly_ignited_coords_for_this_step.begin(), all_newly_ignited_coords_for_this_step.end()),
        all_newly_ignited_coords_for_this_step.end()
    );

    size_t actual_new_cells_this_step = 0;
    for (const auto& coord : all_newly_ignited_coords_for_this_step) {
      // Final check against the global burned_bin to ensure it wasn't
      // already burned in a *previous* time step (the parallel loop's check was against
      // burned_bin's state *at the beginning* of the current time step).
      if (!burned_bin[coord]) { // Check if it's truly new globally
        burned_ids.push_back(coord);
        burned_bin[coord] = true; // Mark it in the global grid
        actual_new_cells_this_step++;
      }
    }

    start = end_of_current_step_processing; // Advance start for the next iteration of the while loop
    // 'end' is implicitly tracked by burned_ids.size()
    current_step_burning_count = actual_new_cells_this_step; // Number of cells to process in the next while loop iteration

    if (actual_new_cells_this_step > 0) {
      burned_ids_steps.push_back(burned_ids.size());
    }

  } // End while(current_step_burning_count > 0)

  return { n_col, n_row, burned_bin, burned_ids, burned_ids_steps };
}