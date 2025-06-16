#include "spread_functions.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <cmath>

// --- CUDA Kernel for a single step of fire spread ---
__global__ void fire_spread_step_kernel(
    Cell* d_landscape_cells,
    bool* d_burned_bin,
    Coord* d_current_burning_ids,
    size_t num_current_burning,
    SimulationParams d_simulation_params,
    float distance, float elevation_mean, float inv_elevation_sd_val, float upper_limit,
    size_t landscape_width, size_t landscape_height,
    Coord* d_next_step_candidates,
    unsigned int* d_candidate_count,
    curandState* d_rand_states
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_current_burning) return;

    curandState local_rand_state = d_rand_states[idx];
    Coord current = d_current_burning_ids[idx];
    const Cell& burning_cell = d_landscape_cells[current.y * landscape_width + current.x];

    constexpr int moves[2][8] = {
        { -1, -1, -1,  0, 0, 1, 1, 1 },
        { -1,  0,  1, -1, 1,-1, 0, 1 }
    };
    constexpr float angles[8] = { 0.0f, 0.78f, 1.57f, 2.35f, 3.14f, 3.92f, 4.71f, 5.5f };

    float inv_dist_val = (distance != 0.0f) ? 1.0f / distance : 0.0f;

    for (int n = 0; n < 8; ++n) {
        int neigh_x = static_cast<int>(current.x) + moves[0][n];
        int neigh_y = static_cast<int>(current.y) + moves[1][n];

        if (neigh_x >= 0 && neigh_x < landscape_width && neigh_y >= 0 && neigh_y < landscape_height) {
            size_t idx_flat = neigh_y * landscape_width + neigh_x;
            if (d_burned_bin[idx_flat] || !d_landscape_cells[idx_flat].burnable) continue;

            const Cell& neighbor = d_landscape_cells[idx_flat];
            float angle = angles[n];

            float elev_diff = neighbor.elevation - burning_cell.elevation;
            float slope_arg = elev_diff * inv_dist_val;
            float slope_term = slope_arg / sqrtf(1.0f + slope_arg * slope_arg);
            float wind_term = cosf(angle - burning_cell.wind_direction);
            float elev_term = (neighbor.elevation - elevation_mean) * inv_elevation_sd_val;

            float linpred = d_simulation_params.independent_pred;
            if (neighbor.vegetation_type == SUBALPINE) linpred += d_simulation_params.subalpine_pred;
            else if (neighbor.vegetation_type == WET)     linpred += d_simulation_params.wet_pred;
            else if (neighbor.vegetation_type == DRY)     linpred += d_simulation_params.dry_pred;

            linpred += d_simulation_params.fwi_pred     * neighbor.fwi;
            linpred += d_simulation_params.aspect_pred  * neighbor.aspect;
            linpred += wind_term                        * d_simulation_params.wind_pred;
            linpred += elev_term                        * d_simulation_params.elevation_pred;
            linpred += slope_term                       * d_simulation_params.slope_pred;

            float prob = upper_limit / (1.0f + expf(-linpred));
            float rand_val = curand_uniform(&local_rand_state);

            if (rand_val < prob) {
                unsigned int candidate_idx = atomicAdd(d_candidate_count, 1);
                if (candidate_idx < landscape_width * landscape_height * 8) {
                    d_next_step_candidates[candidate_idx] = Coord{ static_cast<size_t>(neigh_x), static_cast<size_t>(neigh_y) };
                }
            }
        }
    }

    d_rand_states[idx] = local_rand_state;
}

// --- Host-side simulate_fire function ---
Fire simulate_fire_cuda(
    const Landscape& host_landscape, const std::vector<Coord>& host_ignition_cells,
    SimulationParams host_params, float distance, float elevation_mean, float elevation_sd,
    float upper_limit
) {
    // Allocate GPU Memory
    Cell* d_landscape_cells;
    cudaMalloc(&d_landscape_cells, host_landscape.cells.elems.size() * sizeof(Cell));
    cudaMemcpy(d_landscape_cells, host_landscape.cells.elems.data(),
               host_landscape.cells.elems.size() * sizeof(Cell), cudaMemcpyHostToDevice);

    bool* d_burned_bin;
    size_t bin_size = host_landscape.width * host_landscape.height * sizeof(bool);
    cudaMalloc(&d_burned_bin, bin_size);
    cudaMemset(d_burned_bin, 0, bin_size);

    curandState* d_rand_states;
    // TODO: allocate and init d_rand_states as needed

    std::vector<Coord> host_all_burned_ids = host_ignition_cells;
    Matrix<bool> host_burned_bin(host_landscape.width, host_landscape.height);
    for (const auto& p : host_ignition_cells) host_burned_bin[{p.x, p.y}] = true;

    size_t current_burning_count_host = host_ignition_cells.size();
    std::vector<Coord> host_current_burning_ids = host_ignition_cells;

    Fire result_fire(host_landscape.width, host_landscape.height);
    result_fire.burned_ids_steps.push_back(host_all_burned_ids.size());

    unsigned int* d_candidate_count;
    cudaMalloc(&d_candidate_count, sizeof(unsigned int));
    Coord* d_next_step_candidates;
    size_t max_candidates = host_landscape.width * host_landscape.height * 8;
    cudaMalloc(&d_next_step_candidates, max_candidates * sizeof(Coord));

    while (current_burning_count_host > 0) {
        Coord* d_current_burning_ids_gpu;
        cudaMalloc(&d_current_burning_ids_gpu, current_burning_count_host * sizeof(Coord));
        cudaMemcpy(d_current_burning_ids_gpu, host_current_burning_ids.data(),
                   current_burning_count_host * sizeof(Coord), cudaMemcpyHostToDevice);

        cudaMemset(d_candidate_count, 0, sizeof(unsigned int));

        int threads_per_block = 256;
        int blocks = (current_burning_count_host + threads_per_block - 1) / threads_per_block;

        fire_spread_step_kernel<<<blocks, threads_per_block>>>(
            d_landscape_cells, d_burned_bin, d_current_burning_ids_gpu, current_burning_count_host,
            host_params, distance, elevation_mean, (elevation_sd != 0.0f ? 1.0f / elevation_sd : 0.0f), upper_limit,
            host_landscape.width, host_landscape.height,
            d_next_step_candidates, d_candidate_count, d_rand_states
        );
        cudaDeviceSynchronize();

        cudaFree(d_current_burning_ids_gpu);

        unsigned int num_candidates_host;
        cudaMemcpy(&num_candidates_host, d_candidate_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        if (num_candidates_host == 0) break;

        std::vector<Coord> host_candidates(num_candidates_host);
        cudaMemcpy(host_candidates.data(), d_next_step_candidates,
                   num_candidates_host * sizeof(Coord), cudaMemcpyDeviceToHost);

        std::sort(host_candidates.begin(), host_candidates.end(),
                  [](const Coord& a, const Coord& b) {
                      return (a.y < b.y) || (a.y == b.y && a.x < b.x);
                  });
        host_candidates.erase(std::unique(host_candidates.begin(), host_candidates.end()), host_candidates.end());

        host_current_burning_ids.clear();
        std::vector<Coord> new_burned;

        for (const auto& p : host_candidates) {
            if (!host_burned_bin[{p.x, p.y}]) {
                host_burned_bin[{p.x, p.y}] = true;
                host_all_burned_ids.push_back(p);
                host_current_burning_ids.push_back(p);
                new_burned.push_back(p);
            }
        }

        current_burning_count_host = host_current_burning_ids.size();
        if (current_burning_count_host > 0)
            result_fire.burned_ids_steps.push_back(host_all_burned_ids.size());
    }

    result_fire.burned_ids = std::vector<std::pair<size_t, size_t>>(
        host_all_burned_ids.begin(), host_all_burned_ids.end());

    result_fire.burned_layer = host_burned_bin;

    cudaFree(d_landscape_cells);
    cudaFree(d_burned_bin);
    cudaFree(d_next_step_candidates);
    cudaFree(d_candidate_count);
    cudaFree(d_rand_states);

    return result_fire;
}
