#include "spread_functions.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

// --- CUDA Kernel for setting up cuRAND states ---
__global__ void setup_kernel(curandState *state, unsigned long long seed, size_t num_states) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_states) {
        // Initialize each state with a unique seed and sequence number
        // Using id as the sequence number and a common seed.
        // For more robust randomness across launches, the seed could also vary.
        curand_init(seed, id, 0, &state[id]);
    }
}

// --- CUDA Kernel to mark cells as true based on coordinates (for initial ignitions) ---
__global__ void kernel_mark_cells_as_true(bool* d_bin, const Coord* d_coords_to_mark, size_t num_coords, size_t landscape_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_coords) {
        Coord cell = d_coords_to_mark[idx];
        // Basic bounds check for safety, though coordinates should be valid if generated correctly
        if (cell.x < landscape_width && cell.y < (SIZE_MAX / landscape_width)) { // Avoid overflow with y
             d_bin[cell.y * landscape_width + cell.x] = true;
        }
    }
}

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
    const Landscape& host_landscape, const std::vector<std::pair<size_t, size_t>>& host_ignition_cells,
    SimulationParams host_params, float distance, float elevation_mean, float elevation_sd,
    float upper_limit
) {
    std::vector<Coord> ignition_cells;
    ignition_cells.reserve(host_ignition_cells.size());
    for (const auto& p : host_ignition_cells) {
        ignition_cells.push_back({p.first, p.second});
    }
    // Allocate GPU Memory
    Cell* d_landscape_cells;
    cudaMalloc(&d_landscape_cells, host_landscape.cells.elems.size() * sizeof(Cell));
    cudaMemcpy(d_landscape_cells, host_landscape.cells.elems.data(),
               host_landscape.cells.elems.size() * sizeof(Cell), cudaMemcpyHostToDevice);

    bool* d_burned_bin;
    size_t bin_size = host_landscape.width * host_landscape.height * sizeof(bool);
    cudaMalloc(&d_burned_bin, bin_size);
    cudaMemset(d_burned_bin, 0, bin_size);

    // --- START cuRAND Initialization ---
    curandState* d_rand_states;
    // Number of RNG states: one for each cell in the landscape, as a safe upper bound
    // since 'idx' in fire_spread_step_kernel can go up to num_current_burning.
    // If num_current_burning can be up to total cells, this is appropriate.
    size_t num_rng_states = host_landscape.width * host_landscape.height;
    cudaError_t err = cudaMalloc(&d_rand_states, num_rng_states * sizeof(curandState));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate d_rand_states: %s\n", cudaGetErrorString(err));
        // Handle error appropriately, e.g., return an empty Fire object or throw
        return Fire(0,0); // Example error handling
    }

    int threads_per_block_rng = 1024;
    int blocks_rng = (num_rng_states + threads_per_block_rng - 1) / threads_per_block_rng;
    // Use time(0) or another source for a seed that changes per run
    // For reproducibility during debugging, you might use a fixed seed.
    setup_kernel<<<blocks_rng, threads_per_block_rng>>>(d_rand_states, time(0), num_rng_states);
    err = cudaGetLastError(); // Check for errors in kernel launch
    if (err != cudaSuccess) {
        fprintf(stderr, "setup_kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_rand_states); // Clean up allocated memory
        // Handle error
        return Fire(0,0);
    }
    err = cudaDeviceSynchronize(); // Ensure setup_kernel completes
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize after setup_kernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_rand_states);
        // Handle error
        return Fire(0,0);
    }
    // --- END cuRAND Initialization ---

    std::vector<Coord> host_all_burned_ids = ignition_cells;
    Matrix<bool> host_burned_bin(host_landscape.width, host_landscape.height);
    for (const auto& p : ignition_cells) host_burned_bin[{p.x, p.y}] = true;

    size_t current_burning_count_host = ignition_cells.size();
    std::vector<Coord> host_current_burning_ids = ignition_cells;

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

        int threads_per_block = 1024;
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

// --- CUDA Kernel to accumulate burned cells from multiple simulations ---
__global__ void accumulate_burned_cells_kernel(
    const bool* d_burned_bin_single_sim, // Burned state for one completed simulation
    size_t* d_global_total_burned_counts, // Global accumulator
    size_t width, size_t height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        size_t flat_idx = y * width + x;
        if (d_burned_bin_single_sim[flat_idx]) {
            // atomicAdd for size_t (unsigned long long)
            atomicAdd((unsigned long long int*)&d_global_total_burned_counts[flat_idx], 1ULL);
        }
    }
}

// Internal host function to run one replicate entirely on GPU and accumulate
// Assumes d_landscape_cells, d_all_rand_states, and d_global_total_burned_counts are pre-allocated
__host__ void run_single_replicate_and_accumulate(
    // Globally allocated GPU resources
    Cell* d_landscape_cells,
    SimulationParams sim_params_for_kernel, // <<<< Changed: Pass by value (original was already by value, but name clarified)
    curandState* d_all_rand_states,
    size_t num_rng_states_in_pool,
    size_t* d_global_total_burned_counts,

    // Host-side info for this replicate
    const Landscape& host_landscape_props,
    const std::vector<Coord>& host_initial_ignition_for_this_replicate,
    float distance, float elevation_mean, float elevation_sd, float upper_limit
) {
    // --- 1. Allocate PER-REPLICATE GPU Memory ---
    bool* d_burned_bin_this_sim;
    size_t bin_size = host_landscape_props.width * host_landscape_props.height * sizeof(bool);
    cudaMalloc(&d_burned_bin_this_sim, bin_size);
    cudaMemset(d_burned_bin_this_sim, 0, bin_size); // Reset for this simulation

    // Initialize d_burned_bin_this_sim with ignition points for this replicate
    // This requires copying host_initial_ignition_for_this_replicate to GPU and launching a small kernel
    Coord* d_initial_ignitions_this_sim;
    cudaMalloc(&d_initial_ignitions_this_sim, host_initial_ignition_for_this_replicate.size() * sizeof(Coord));
    cudaMemcpy(d_initial_ignitions_this_sim, host_initial_ignition_for_this_replicate.data(),
               host_initial_ignition_for_this_replicate.size() * sizeof(Coord), cudaMemcpyHostToDevice);

    // --- Mark initial ignitions in d_burned_bin_this_sim ---
    if (!host_initial_ignition_for_this_replicate.empty()) {
        int threads_init_mark = 1024;
        int blocks_init_mark = (host_initial_ignition_for_this_replicate.size() + threads_init_mark - 1) / threads_init_mark;
        if (host_initial_ignition_for_this_replicate.size() > 0 && blocks_init_mark > 0) {
            kernel_mark_cells_as_true<<<blocks_init_mark, threads_init_mark>>>(
                d_burned_bin_this_sim,
                d_initial_ignitions_this_sim,
                host_initial_ignition_for_this_replicate.size(),
                host_landscape_props.width
            );
            cudaError_t err_init_mark = cudaGetLastError();
            if (err_init_mark != cudaSuccess) {
                fprintf(stderr, "Initial kernel_mark_cells_as_true launch failed: %s\n", cudaGetErrorString(err_init_mark));
            }
            cudaDeviceSynchronize(); // Ensure initial state is set
        }
    }
    // --- End initial marking ---

    std::vector<Coord> host_current_burning_ids = host_initial_ignition_for_this_replicate;
    size_t current_burning_count_host = host_current_burning_ids.size(); // Initialize count

    unsigned int* d_candidate_count; // Per-replicate
    cudaMalloc(&d_candidate_count, sizeof(unsigned int));
    Coord* d_next_step_candidates; // Per-replicate
    size_t max_candidates = host_landscape_props.width * host_landscape_props.height * 8;
    cudaMalloc(&d_next_step_candidates, max_candidates * sizeof(Coord));

    // --- 2. Fire Spread Loop for THIS Replicate ---
    // Use a device pointer for the current burning IDs. We will swap pointers instead of copying memory.
    Coord* d_current_burning_ids_gpu;
    cudaMalloc(&d_current_burning_ids_gpu, max_candidates * sizeof(Coord)); // Allocate once to max size
    cudaMemcpy(d_current_burning_ids_gpu, host_initial_ignition_for_this_replicate.data(),
               host_initial_ignition_for_this_replicate.size() * sizeof(Coord), cudaMemcpyHostToDevice);

    while (current_burning_count_host > 0) {
        cudaMemset(d_candidate_count, 0, sizeof(unsigned int));

        int threads_per_block = 1024;
        int blocks = (current_burning_count_host + threads_per_block - 1) / threads_per_block;

        fire_spread_step_kernel<<<blocks, threads_per_block>>>(
            d_landscape_cells, d_burned_bin_this_sim, d_current_burning_ids_gpu, current_burning_count_host,
            sim_params_for_kernel,
            distance, elevation_mean, (elevation_sd != 0.0f ? 1.0f / elevation_sd : 0.0f), upper_limit,
            host_landscape_props.width, host_landscape_props.height,
            d_next_step_candidates, d_candidate_count, d_all_rand_states
        );
        // NO cudaDeviceSynchronize() HERE

        unsigned int num_candidates_host;
        // The only D->H copy needed inside the loop is this tiny one to check the count
        cudaMemcpy(&num_candidates_host, d_candidate_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        if (num_candidates_host == 0) break;

        // --- Perform Sort and Unique on GPU using Thrust ---
        // The thrust device_ptr wraps a raw pointer so thrust algorithms can use it.
        thrust::device_ptr<Coord> d_next_step_ptr = thrust::device_pointer_cast(d_next_step_candidates);
        
        // 1. Sort the candidates on the GPU
        thrust::sort(d_next_step_ptr, d_next_step_ptr + num_candidates_host,
                     [] __device__ (const Coord& a, const Coord& b) {
                         return (a.y < b.y) || (a.y == b.y && a.x < b.x);
                     });

        // 2. Find unique elements on the GPU. `new_end` will be a pointer to the end of the unique range.
        thrust::device_ptr<Coord> new_end = thrust::unique(d_next_step_ptr, d_next_step_ptr + num_candidates_host);
        
        // 3. Calculate the number of unique new candidates
        size_t num_unique_candidates = new_end - d_next_step_ptr;
        
        if (num_unique_candidates == 0) break;

        // --- Update d_burned_bin_this_sim with the unique candidates ---
        int threads_mark = 1024;
        int blocks_mark = (num_unique_candidates + threads_mark - 1) / threads_mark;
        if (blocks_mark > 0) {
            // We can use d_next_step_candidates directly as it now holds the unique values at the beginning
            kernel_mark_cells_as_true<<<blocks_mark, threads_mark>>>(
                d_burned_bin_this_sim, 
                d_next_step_candidates, // Use the buffer that now contains the unique sorted candidates
                num_unique_candidates, 
                host_landscape_props.width
            );
        }

        // --- Prepare for next iteration ---
        // Instead of allocating a new buffer, just copy the unique candidates into the input buffer for the next step.
        cudaMemcpy(d_current_burning_ids_gpu, d_next_step_candidates, num_unique_candidates * sizeof(Coord), cudaMemcpyDeviceToDevice);
        current_burning_count_host = num_unique_candidates;
    }
    // --- End Fire Spread Loop ---
    cudaFree(d_current_burning_ids_gpu); // Free the buffer allocated once at the start of the function

    // --- 3. Accumulate results from d_burned_bin_this_sim into d_global_total_burned_counts ---
    dim3 blockDimAcc(16, 16);
    dim3 gridDimAcc(
        (host_landscape_props.width + blockDimAcc.x - 1) / blockDimAcc.x,
        (host_landscape_props.height + blockDimAcc.y - 1) / blockDimAcc.y
    );
    accumulate_burned_cells_kernel<<<gridDimAcc, blockDimAcc>>>(
        d_burned_bin_this_sim, d_global_total_burned_counts,
        host_landscape_props.width, host_landscape_props.height
    );
    cudaDeviceSynchronize(); // Ensure accumulation for this replicate is done

    // --- 4. Free PER-REPLICATE GPU Memory ---
    cudaFree(d_initial_ignitions_this_sim);
    cudaFree(d_burned_bin_this_sim);
    cudaFree(d_next_step_candidates);
    cudaFree(d_candidate_count);
}

// This is the new function that will be called from many_simulations.cpp (via a header declaration)
// It replaces the old simulate_fire_cuda in terms of being the top-level orchestrator for one set of replicates.
// The original simulate_fire_cuda is effectively gone or merged into run_single_replicate_and_accumulate.
Matrix<size_t> burned_amounts_per_cell_on_gpu( // New name for clarity
    const Landscape& host_landscape,
    const std::vector<std::pair<size_t, size_t>>& host_initial_ignition_cells_template, // Template for all sims
    SimulationParams host_sim_params,
    float distance, float elevation_mean, float elevation_sd, float upper_limit,
    size_t n_replicates
) {
    // --- 1. Allocate GLOBAL GPU Resources (Once) ---
    Cell* d_landscape_cells;
    cudaMalloc(&d_landscape_cells, host_landscape.cells.elems.size() * sizeof(Cell));
    cudaMemcpy(d_landscape_cells, host_landscape.cells.elems.data(),
               host_landscape.cells.elems.size() * sizeof(Cell), cudaMemcpyHostToDevice);

    // REMOVE GPU allocation for SimulationParams
    // SimulationParams* d_simulation_params_gpu; 
    // cudaMalloc(&d_simulation_params_gpu, sizeof(SimulationParams));
    // cudaMemcpy(d_simulation_params_gpu, &host_sim_params, sizeof(SimulationParams), cudaMemcpyHostToDevice);

    size_t* d_global_total_burned_counts;
    size_t total_counts_size_bytes = host_landscape.width * host_landscape.height * sizeof(size_t);
    cudaMalloc(&d_global_total_burned_counts, total_counts_size_bytes);
    cudaMemset(d_global_total_burned_counts, 0, total_counts_size_bytes);

    curandState* d_all_rand_states;
    size_t num_rng_states = host_landscape.width * host_landscape.height;
    cudaMalloc(&d_all_rand_states, num_rng_states * sizeof(curandState));
    int threads_rng = 1024;
    int blocks_rng = (num_rng_states + threads_rng - 1) / threads_rng;
    setup_kernel<<<blocks_rng, threads_rng>>>(d_all_rand_states, time(0), num_rng_states);
    cudaDeviceSynchronize();

    std::vector<Coord> host_ignition_coords_template;
    host_ignition_coords_template.reserve(host_initial_ignition_cells_template.size());
    for(const auto& p : host_initial_ignition_cells_template) {
        host_ignition_coords_template.push_back({p.first, p.second});
    }

    // --- 2. Loop n_replicates (on host, orchestrating GPU work) ---
    for (size_t i = 0; i < n_replicates; ++i) {
        run_single_replicate_and_accumulate(
            d_landscape_cells, 
            host_sim_params, // <<<< Pass the host-side struct directly
            d_all_rand_states, num_rng_states, d_global_total_burned_counts,
            host_landscape, host_ignition_coords_template,
            distance, elevation_mean, elevation_sd, upper_limit
        );
    }

    // --- 3. Copy Final Aggregated Results Back to Host ---
    Matrix<size_t> host_final_burned_amounts(host_landscape.width, host_landscape.height);
    cudaMemcpy(host_final_burned_amounts.elems.data(), d_global_total_burned_counts,
               total_counts_size_bytes, cudaMemcpyDeviceToHost);

    // --- 4. Free GLOBAL GPU Resources ---
    cudaFree(d_landscape_cells);
    // cudaFree(d_simulation_params_gpu); // REMOVE this free
    cudaFree(d_global_total_burned_counts);
    cudaFree(d_all_rand_states);

    return host_final_burned_amounts;
}
