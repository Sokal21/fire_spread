#include "many_simulations.hpp"

// spread_functions.hpp is included via many_simulations.hpp, ensure it has the new declaration

Matrix<size_t> burned_amounts_per_cell(
    const Landscape& landscape, const std::vector<std::pair<size_t, size_t>>& ignition_cells,
    SimulationParams params, float distance, float elevation_mean, float elevation_sd,
    float upper_limit, size_t n_replicates
) {
    // Call the new GPU orchestrator function
    return burned_amounts_per_cell_on_gpu(
        landscape, ignition_cells, params, distance, elevation_mean, elevation_sd, upper_limit, n_replicates
    );
}
