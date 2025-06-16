// filepath: /home/tomas/Desktop/fire_spread/src/spread_functions.cu
#include "spread_functions.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h> // For cuRAND device functions
#include <device_launch_parameters.h> // For <<< >>>
#include <algorithm> // For std::sort, std::unique on host

// #include "random_pool.hpp" // CPU version, not for GPU kernel

#define _USE_MATH_DEFINES
#include <cmath>
// <immintrin.h> is for CPU AVX, not needed in kernel directly
// <omp.h> is for CPU OpenMP, not for kernel

// Ensure Cell and SimulationParams are "Plain Old Data" (POD) or suitable for GPU
// (no virtual functions, complex C++ features not well supported in kernels without care)

// --- CUDA Kernel for a single step of fire spread ---
__global__ void fire_spread_step_kernel(
    Cell* d_landscape_cells,
    bool* d_burned_bin, // Global state of all burned cells
    std::pair<size_t, size_t>* d_current_burning_ids,
    size_t num_current_burning,
    SimulationParams d_simulation_params,
    float distance, float elevation_mean, float inv_elevation_sd_val, float upper_limit,
    size_t landscape_width, size_t landscape_height,
    std::pair<size_t, size_t>* d_next_step_candidates, // Output buffer for potential new ignitions
    unsigned int* d_candidate_count, // Atomic counter for d_next_step_candidates
    curandState* d_rand_states
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_current_burning) return;

    curandState local_rand_state = d_rand_states[idx]; // Load thread's RNG state

    std::pair<size_t, size_t> current_burn_coord = d_current_burning_ids[idx];
    const Cell& burning_cell = d_landscape_cells[current_burn_coord.second * landscape_width + current_burn_coord.first];

    constexpr int moves[2][8] = { { -1, -1, -1, 0, 0, 1, 1, 1 },
                                  { -1, 0, 1, -1, 1, -1, 0, 1 } };
    constexpr float angles[8] = { /* ... your angles ... */ }; // Ensure M_PI is defined or use literal

    float inv_dist_val = (distance != 0.0f) ? 1.0f / distance : 0.0f;

    for (int n = 0; n < 8; ++n) {
        int neigh_x = static_cast<int>(current_burn_coord.first) + moves[0][n];
        int neigh_y = static_cast<int>(current_burn_coord.second) + moves[1][n];

        if (neigh_x >= 0 && neigh_x < landscape_width && neigh_y >= 0 && neigh_y < landscape_height) {
            size_t neigh_flat_idx = neigh_y * landscape_width + neigh_x;
            if (d_burned_bin[neigh_flat_idx] || !d_landscape_cells[neigh_flat_idx].burnable) {
                continue;
            }

            const Cell& neighbour_cell = d_landscape_cells[neigh_flat_idx];
            float angle = angles[n];

            // --- Scalar probability calculation (GPU threads execute this in parallel) ---
            // Replicate the logic from spread_probability_scalar or your AVX version using scalar math
            // Math functions like cosf, expf, sqrtf are available in device code.

            float elev_diff = neighbour_cell.elevation - burning_cell.elevation;
            float slope_arg = elev_diff * inv_dist_val; // X for slope term
            float slope_term = slope_arg / sqrtf(1.0f + slope_arg * slope_arg); // sin(atan(X))

            float wind_term = cosf(angle - burning_cell.wind_direction);
            float elev_term = (neighbour_cell.elevation - elevation_mean) * inv_elevation_sd_val;
            float linpred = d_simulation_params.independent_pred;

            if (neighbour_cell.vegetation_type == SUBALPINE) linpred += d_simulation_params.subalpine_pred;
            else if (neighbour_cell.vegetation_type == WET) linpred += d_simulation_params.wet_pred;
            else if (neighbour_cell.vegetation_type == DRY) linpred += d_simulation_params.dry_pred;

            linpred += d_simulation_params.fwi_pred * neighbour_cell.fwi;
            linpred += d_simulation_params.aspect_pred * neighbour_cell.aspect;
            linpred += wind_term * d_simulation_params.wind_pred;
            linpred += elev_term * d_simulation_params.elevation_pred;
            linpred += slope_term * d_simulation_params.slope_pred;

            float prob = upper_limit / (1.0f + expf(-linpred));
            // --- End scalar probability calculation ---

            float rand_val = curand_uniform(&local_rand_state);

            if (rand_val < prob) {
                unsigned int candidate_idx = atomicAdd(d_candidate_count, 1);
                // Ensure d_next_step_candidates is large enough
                if (candidate_idx < landscape_width * landscape_height * 8) { // Max possible candidates
                     d_next_step_candidates[candidate_idx] = {static_cast<size_t>(neigh_x), static_cast<size_t>(neigh_y)};
                }
            }
        }
    }
    d_rand_states[idx] = local_rand_state; // Store back RNG state
}

// --- Host-side simulate_fire function ---
Fire simulate_fire_cuda(
    const Landscape& host_landscape, const std::vector<std::pair<size_t, size_t>>& host_ignition_cells,
    SimulationParams host_params, float distance, float elevation_mean, float elevation_sd,
    float upper_limit
) {
    // 0. CUDA Device Initialization (error checking omitted for brevity)
    // cudaSetDevice(0);

    // 1. Allocate GPU Memory
    Cell* d_landscape_cells;
    cudaMalloc(&d_landscape_cells, host_landscape.cells.elems.size() * sizeof(Cell));
    // ... d_burned_bin, d_current_burning_ids, d_next_step_candidates, d_candidate_count, d_rand_states

    bool* d_burned_bin;
    size_t bin_size = host_landscape.width * host_landscape.height * sizeof(bool);
    cudaMalloc(&d_burned_bin, bin_size);
    cudaMemset(d_burned_bin, 0, bin_size);
    // 2. Copy Data Host to Device
    cudaMemcpy(d_landscape_cells, host_landscape.cells.elems.data(), host_landscape.cells.elems.size() * sizeof(Cell), cudaMemcpyHostToDevice);
    // ... copy initial burned_bin, initial ignition_cells to d_current_burning_ids, params

    // Initialize cuRAND states (example, more robust initialization needed)
    curandState* d_rand_states;
    // ... cudaMalloc for d_rand_states ...
    // Kernel to initialize cuRAND states for each potential thread:
    // setup_kernel<<<blocks, threads_per_block>>>(d_rand_states, time(0), num_max_threads_or_burning_cells);

    std::vector<std::pair<size_t, size_t>> host_all_burned_ids = host_ignition_cells;
    Matrix<bool> host_burned_bin(host_landscape.width, host_landscape.height); // CPU copy
    for(const auto& p : host_ignition_cells) host_burned_bin[p] = true;
    // ... copy host_burned_bin to d_burned_bin ...

    size_t current_burning_count_host = host_ignition_cells.size();
    std::vector<std::pair<size_t, size_t>> host_current_burning_ids = host_ignition_cells;

    Fire result_fire(host_landscape.width, host_landscape.height);
    result_fire.burned_ids_steps.push_back(host_all_burned_ids.size());


    unsigned int* d_candidate_count;
    cudaMalloc(&d_candidate_count, sizeof(unsigned int));
    std::pair<size_t, size_t>* d_next_step_candidates;
    // Max possible candidates: all cells * 8 neighbors (overestimation, but safe for buffer)
    size_t max_candidates = host_landscape.width * host_landscape.height * 8;
    cudaMalloc(&d_next_step_candidates, max_candidates * sizeof(std::pair<size_t, size_t>));


    while (current_burning_count_host > 0) {
        // Copy current burning IDs to device
        std::pair<size_t, size_t>* d_current_burning_ids_gpu;
        cudaMalloc(&d_current_burning_ids_gpu, current_burning_count_host * sizeof(std::pair<size_t, size_t>));
        cudaMemcpy(d_current_burning_ids_gpu, host_current_burning_ids.data(), current_burning_count_host * sizeof(std::pair<size_t, size_t>), cudaMemcpyHostToDevice);

        // Reset candidate count on device
        cudaMemset(d_candidate_count, 0, sizeof(unsigned int));

        // 3. Kernel Launch Configuration
        int threads_per_block = 256;
        int blocks = (current_burning_count_host + threads_per_block - 1) / threads_per_block;

        fire_spread_step_kernel<<<blocks, threads_per_block>>>(
            d_landscape_cells, d_burned_bin, d_current_burning_ids_gpu, current_burning_count_host,
            host_params, distance, elevation_mean, (elevation_sd != 0.0f ? 1.0f/elevation_sd : 0.0f), upper_limit,
            host_landscape.width, host_landscape.height,
            d_next_step_candidates, d_candidate_count, d_rand_states
        );
        cudaDeviceSynchronize(); // Wait for kernel to finish

        cudaFree(d_current_burning_ids_gpu); // Free per-step buffer

        // 4. Process results (d_next_step_candidates)
        unsigned int num_candidates_host;
        cudaMemcpy(&num_candidates_host, d_candidate_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        if (num_candidates_host == 0) {
            current_burning_count_host = 0;
            break;
        }

        std::vector<std::pair<size_t, size_t>> host_candidates(num_candidates_host);
        cudaMemcpy(host_candidates.data(), d_next_step_candidates, num_candidates_host * sizeof(std::pair<size_t, size_t>), cudaMemcpyDeviceToHost);

        // Sort and unique on CPU (for simplicity here, can be done on GPU with Thrust)
        std::sort(host_candidates.begin(), host_candidates.end());
        host_candidates.erase(std::unique(host_candidates.begin(), host_candidates.end()), host_candidates.end());

        host_current_burning_ids.clear();
        std::vector<std::pair<size_t, size_t>> new_cells_for_d_burned_bin_update;

        for (const auto& p : host_candidates) {
            if (!host_burned_bin[p]) { // Check against CPU's master burned_bin
                host_burned_bin[p] = true;
                host_all_burned_ids.push_back(p);
                host_current_burning_ids.push_back(p); // These are the ones for the next GPU step
                new_cells_for_d_burned_bin_update.push_back(p);
            }
        }
        current_burning_count_host = host_current_burning_ids.size();

        // Update d_burned_bin on GPU (can be done with another small kernel)
        // For each cell in new_cells_for_d_burned_bin_update, set d_burned_bin[coord] = true;
        // This is simplified; a kernel would be more efficient for large updates.
        if (!new_cells_for_d_burned_bin_update.empty()) {
             // Example: copy updated host_burned_bin back, or write a kernel to update d_burned_bin
             // For now, let's assume d_burned_bin is updated based on host_burned_bin if needed for next kernel pass
             // A more efficient way is a kernel that takes new_cells_for_d_burned_bin_update and updates d_burned_bin
        }


        if (current_burning_count_host > 0) {
             result_fire.burned_ids_steps.push_back(host_all_burned_ids.size());
        }
    }

    // 5. Copy final results back (if needed, e.g. final burned_ids)
    result_fire.burned_ids = host_all_burned_ids;
    // Copy final host_burned_bin to result_fire.burned_layer
    result_fire.burned_layer = host_burned_bin;


    // 6. Free GPU Memory
    cudaFree(d_landscape_cells);
    cudaFree(d_burned_bin);
    cudaFree(d_next_step_candidates);
    cudaFree(d_candidate_count);
    cudaFree(d_rand_states);
    // ... free other allocations ...

    return result_fire;
}