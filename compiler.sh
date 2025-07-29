#!/bin/bash
if [ $# -lt 3 ]; then
    echo "Usage: $0 <compiler> <compiler_flags> <iterations>"
    exit 1
fi

ITERATIONS=$3
# Clean first
make clean

# Time the compilation and execution
make COMPILER=$1 MORE_CXXFLAGS="$2"

for ((i=1; i<=$ITERATIONS; i++)); do
    echo "Running iteration $i of $ITERATIONS"
# Run AMDuProf analysis
    AMDuProfCLI collect --event RETIRED_INST --event RETIRED_SSE_AVX_FLOPS --event CYCLES_NOT_IN_HALT \
            --interval 1 -o amdprof_results \
            ./graphics/burned_probabilities_data ./data/1999_27j_S > /dev/null 2>&1
done

# Create reports directory if it doesn't exist
mkdir -p reports

timestamp=$(date +%Y%m%d_%H%M%S)
# Create CSV header in reports folder
echo "Iteration,Duration,Cycles,Instructions,FP_Operations,IPC,IPS,FLOPS" > "reports/results_$1_$timestamp.csv"

# Generate AMD uProf reports for all profile directories
for profile_dir in amdprof_results/AMDuProf-burned_probabilities_data-Custom_*; do
    echo "Generating report for: $profile_dir"
    AMDuProfCLI report -i "$profile_dir"
    
    echo "Extracting metrics from: $profile_dir"
    # Get duration from the report file
    duration=$(grep "Profile Duration" "$profile_dir/report.csv" | cut -d'"' -f2 | cut -d' ' -f1)
    
    # Get the metrics from the HOTTEST PROCESSES section
    metrics=$(grep "burned_probabil (PID:" "$profile_dir/report.csv" | sed -n '2p')
    cycles=$(echo "$metrics" | cut -d',' -f3 | tr -d '"' | tr -d ' ')
    instructions=$(echo "$metrics" | cut -d',' -f4 | tr -d '"' | tr -d ' ')
    flops_count=$(echo "$metrics" | cut -d',' -f5 | tr -d '"' | tr -d ' ')

    ipc=$(echo "scale=4; $instructions / $cycles" | bc)
    ips=$(echo "scale=4; $instructions / $duration" | bc)
    flops=$(echo "scale=4; $flops_count / $duration" | bc)
    
    # Append results to CSV in reports folder
    echo "$profile_dir,$duration,$cycles,$instructions,$flops_count,$ipc,$ips,$flops" >> "reports/results_$1_$timestamp.csv"
    
    echo "Results for $profile_dir:"
    echo "IPC (Instructions per Cycle): $ipc"
    echo "IPS (Instructions per Second): $ips"
    echo "FLOPS (Floating Point Operations per Second): $flops"
    echo "Raw counts:"
    echo "  Cycles: $cycles"
    echo "  Instructions: $instructions"
    echo "  FP Operations: $flops_count"
    echo "  Duration (s): $duration"
    echo "-------------------"
done

# Clean up AMD uProf results
rm -rf amdprof_results/*

# You can still keep the hyperfine benchmark if needed
# hyperfine --warmup 2 --runs 5 './graphics/fire_animation_data ./data/2000_8'
