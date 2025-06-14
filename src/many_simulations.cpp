#include "many_simulations.hpp"

#include <cmath>
#include <omp.h> // Include OpenMP header

Matrix<size_t> burned_amounts_per_cell(
    const Landscape& landscape, const std::vector<std::pair<size_t, size_t>>& ignition_cells,
    SimulationParams params, float distance, float elevation_mean, float elevation_sd,
    float upper_limit, size_t n_replicates
) {
    Matrix<size_t> host_burned_amounts(landscape.width, landscape.height);
    // Initialize to 0

    // Option 1: Run sequentially on CPU, but each simulate_fire is GPU accelerated
    // #pragma omp parallel for schedule(dynamic) // Can still use OpenMP for CPU-side replicate management
    for (size_t i = 0; i < n_replicates; i++) {
        Fire fire = simulate_fire_cuda( // Call the CUDA version
            landscape, ignition_cells, params, distance, elevation_mean, elevation_sd, upper_limit
        );
        // Aggregate results on CPU (critical section if OpenMP is used here)
        // #pragma omp critical
        for (const auto& burned_cell_coords : fire.burned_ids) {
            if (burned_cell_coords.first < landscape.width && burned_cell_coords.second < landscape.height) {
                host_burned_amounts[burned_cell_coords] += 1;
            }
        }
    }
    return host_burned_amounts;

    // Option 2: More advanced - try to run multiple simulations concurrently on GPU
    // This would involve managing multiple simulation states on the GPU, much more complex.
    // Or, if n_replicates is very large, a GPU kernel could sum up results.
    // For now, Option 1 is more straightforward.
}
