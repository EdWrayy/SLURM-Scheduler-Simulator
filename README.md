# SLURM-Scheduler-Simulator
This simulator is designed to experiment with the impact on energy usage of different placement heuristics for the SLURM scheduler.
The scope is limited to modelling the spatial placement of jobs; this means it only chooses which nodes and hardware jobs run on, not when they run.
It is designed to process real accounting logs to evaluate different heuristics under different power model assumptions.
Analysis compares the tradeoff between energy usage and schedule disruption.



## Project Structure

This project consists of 5 main components:

### 1. Slurm Simulator
Handles running the simulation, configurable with a power model, scheduling strategies, SLURM configuration, and jobs to schedule.

### 2. Data Handling
Responsible for converting SLURM accounting logs into a parquet file formatted for input into the simulator.

### 3. Simulation Analysis
Takes the results of a single simulation run as input, computes key metrics for analysis, and generates plots for visual aid. These include:

**Energy Metrics:**
- Total Energy (kWh)
- Average Power (kW)
- Energy Saving vs Baseline (%)

**Schedule Preservation Metrics:**
- Dropped Jobs (%)
- Dropped Work (%)

**Node Consolidation Metrics:**
- Mean Active Nodes
- Peak Active Nodes
- Node-Hours Active

**Utilisation Metrics:**
- CPU Utilisation (mean and p95)
- GPU Utilisation (mean and p95)
- Memory Utilisation (mean and p95)

**Fragmentation Metrics:**
- Mean free CPUs on active nodes
- Mean free GPUs on active nodes
- Mean free memory on active nodes

**Plots:**
- Power over time
- Active Nodes Over Time
- Dropped Work Composition - for dropped jobs, shows which resource was lacking which prevented it from being scheduled. This provides insight into which workload categories are most affected.


### 4. Meta Analysis
This takes the reports from many different simulations and provides useful metrics for comparison. 
This includes:

**Pareto Trade-Off Plot:** A scatter plot with:
- x-axis: Dropped Work (%)
- y-axis: Energy Saving (%)

For each metric computed in the simulation analysis, meta analysis ranks them from best to worst.

### 5. Common
Data structures used by multiple packages.
