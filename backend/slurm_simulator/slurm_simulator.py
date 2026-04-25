from backend.common.models import Job
import math
from datetime import datetime
from itertools import combinations


class Node:
    def __init__(self, name, node_type, id, list_position, physical_CPUs, cores_per_physical_CPU, total_CPUs, total_GPUs, total_memory, CPU_Max_Power, GPU_Max_Power, RAM_Power, power_model):
        self.name = name
        self.node_type = node_type
        self.id = id 
        self.list_position = list_position #The position in the slurm.conf listing, relevant for resource distirbution.

        self.physical_CPUs = physical_CPUs
        self.cores_per_physical_CPU = cores_per_physical_CPU
        self.total_CPUs = total_CPUs
        self.total_GPUs = total_GPUs
        self.total_memory = total_memory

        self.CPUs_in_use = 0
        self.GPUs_in_use = 0
        self.memory_in_use = 0

        self.CPU_MAX_Power_Consumption = CPU_Max_Power
        self.GPU_MAX_Power_Consumption = GPU_Max_Power
        self.RAM_Power_Consumption = RAM_Power

        self.current_power_consumption = 0
        self.power_model = power_model
        self._refresh_power()

    def _refresh_power(self) -> None:
        self.current_power_consumption = self.power_model.calculate_current_power_consumption(self)


    def run_job(self, CPUs_required, GPUs_required, memory_required):
        if self.CPUs_in_use + CPUs_required > self.total_CPUs:
            raise ValueError(f"CPU overallocation on {self.name}")
        if self.GPUs_in_use + GPUs_required > self.total_GPUs:
            raise ValueError(f"GPU overallocation on {self.name}")
        if self.memory_in_use + memory_required > self.total_memory:
            raise ValueError(f"Memory overallocation on {self.name}")

        self.CPUs_in_use += CPUs_required
        self.GPUs_in_use += GPUs_required
        self.memory_in_use += memory_required
        self._refresh_power()


    def release_job(self, CPUs_required, GPUs_required, memory_required):
        self.CPUs_in_use -= CPUs_required
        self.GPUs_in_use -= GPUs_required
        self.memory_in_use -= memory_required
        self._refresh_power()



class NodePowerModel:
    """
    Base class for power model.
    Determines how to compute a node's current power utilisation.
    """
    def calculate_current_power_consumption(self, node):
        raise NotImplementedError


class ActiveOnlyPowerModel(NodePowerModel):
    """Simply assumes we only use power for the exact hardware used, no idling, no powerdown."""

    def calculate_current_power_consumption(self, node):
        cpu_u = math.ceil(node.CPUs_in_use / node.cores_per_physical_CPU) * node.CPU_MAX_Power_Consumption
        gpu_u = node.GPUs_in_use * node.GPU_MAX_Power_Consumption

        return cpu_u + gpu_u + node.RAM_Power_Consumption
    
class LinearWithIdlePowerModel(NodePowerModel):
    """
    Assumes that all hardware is either in use or idling if a node is powered on.
    Cannot power down any nodes
    Although energy savings cannot be made by powering down nodes, they can still be made by prioritising more energy efficient nodes.
    """
    def __init__(self, cpu_idle_fraction, gpu_idle_fraction):
        self.cpu_idle_fraction = cpu_idle_fraction
        self.gpu_idle_fraction = gpu_idle_fraction

    def calculate_current_power_consumption(self, node):
        cpu_u = math.ceil(node.CPUs_in_use / node.cores_per_physical_CPU) * node.CPU_MAX_Power_Consumption
        cpu_idle =  math.ceil((node.total_CPUs - node.CPUs_in_use) / node.cores_per_physical_CPU) * node.CPU_MAX_Power_Consumption * self.cpu_idle_fraction

        gpu_u = node.GPUs_in_use * node.GPU_MAX_Power_Consumption
        gpu_idle = (node.total_GPUs - node.GPUs_in_use) *  node.GPU_MAX_Power_Consumption  * self.gpu_idle_fraction

        return cpu_u + gpu_u + cpu_idle + gpu_idle + node.RAM_Power_Consumption
    

class LinearWithSleepPowerModel(NodePowerModel):
    """
    Assumes that all hardware is either in use or idling if a node is powered on.
    Can sleep an entire node if no jobs running on it (using no power)
    """
    def __init__(self, cpu_idle_fraction, gpu_idle_fraction):
        self.cpu_idle_fraction = cpu_idle_fraction
        self.gpu_idle_fraction = gpu_idle_fraction

    def calculate_current_power_consumption(self, node):

        if node.CPUs_in_use == 0 and node.GPUs_in_use == 0 and node.memory_in_use == 0:
            return 0
        
        cpu_u = math.ceil(node.CPUs_in_use / node.cores_per_physical_CPU) * node.CPU_MAX_Power_Consumption
        cpu_idle =  math.ceil((node.total_CPUs - node.CPUs_in_use) / node.cores_per_physical_CPU) * node.CPU_MAX_Power_Consumption  * self.cpu_idle_fraction

        gpu_u = node.GPUs_in_use * node.GPU_MAX_Power_Consumption
        gpu_idle = (node.total_GPUs - node.GPUs_in_use) *  node.GPU_MAX_Power_Consumption  * self.gpu_idle_fraction

        return cpu_u + gpu_u  + cpu_idle + gpu_idle + node.RAM_Power_Consumption
    


class NodeSelectionStrategy:
    """Base class for node selection strategies when placing jobs"""

    DEFAULT_WEIGHTS = (1.0, 1.0, 1.0)
    MAX_FULL_SUBSET_EVALUATIONS = 5000

    def __init__(self, family_weights: dict | None = None):
        """
        family_weights: optional per-node-type resource weights
        not needed for other heuristics
        """

        self.family_weights: dict[str, tuple[float, float, float]] = {}
        for node_type, w in (family_weights or {}).items():
            if isinstance(w, dict):
                self.family_weights[node_type] = (w["cpu"], w["gpu"], w["memory"])
            else:
                self.family_weights[node_type] = tuple(w)


    
    def _get_weights(self, node) -> tuple[float, float, float]:
        return self.family_weights.get(node.node_type, self.DEFAULT_WEIGHTS)


    def get_free_capacities(self, node):
        return (
            node.total_CPUs - node.CPUs_in_use,
            node.total_GPUs - node.GPUs_in_use,
            node.total_memory - node.memory_in_use,
        )

    
    def nodes_have_capacity(self, nodes, job):
        total_c = sum(self.get_free_capacities(nd)[0] for nd in nodes)
        total_g = sum(self.get_free_capacities(nd)[1] for nd in nodes)
        total_m = sum(self.get_free_capacities(nd)[2] for nd in nodes)

        return (
            total_c >= job.CPUs_required and
            total_g >= job.GPUs_required and
            total_m >= job.memory_required
        )

    
    def build_failure_info(self):
        return {
            "failed": True,
            "reason": "exhaustive_search_did_not_find_feasible_set",
        }


    def get_candidate_node_groups(self, job, island_groups):
        candidate_node_groups = []
        for allowed_type in job.allowed_node_types:
            candidate_node_groups.extend(island_groups[allowed_type])
        return candidate_node_groups


    def _count_subsets(self, pool_size, subset_size):
        if subset_size < 0 or pool_size < subset_size:
            return 0
        return math.comb(pool_size, subset_size)


    def score_single_node(self, node, job) -> float:
        """
        Strategy-specific node pruning score for reducing combinatorial search
        Default is to score subset with single node
        Should be overrided for each heuristic
        """
        return self.score_subset([node], job)


    def _shrink_candidates_to_threshold(self, candidates, job, n):
        reduced = list(candidates)

        while (
            len(reduced) > n and
            self._count_subsets(len(reduced), n) > self.MAX_FULL_SUBSET_EVALUATIONS
        ):
            worst_index = min(
                range(len(reduced)),
                key=lambda i: (
                    self.score_single_node(reduced[i], job),
                    reduced[i].list_position,
                    reduced[i].id,
                ),
            )
            reduced.pop(worst_index)

        return reduced


    def score_subset(self, nodes, job) -> float:
        raise NotImplementedError

    def _evaluate_subset(self, nodes, job) -> float:
        record = self.resource_distribution_strategy.allocate_resources(job, nodes)
        try:
            return self.score_subset(nodes, job)
        finally:
            for node, (cpus, gpus, mem) in record.items():
                node.release_job(cpus, gpus, mem)

    def select_nodes(self, job, island_groups):
        if job.nodes_required <= 0:
            return [], {"failed": True, "reason": "invalid_nodes_required"}
        best_subset = self.select_best_scored_subset(job, island_groups)
        if best_subset is not None:
            return best_subset, {"failed": False, "reason": "scored_best_subset"}
        return [], self.build_failure_info()

    def select_best_scored_subset(self, job, island_groups):
        n = job.nodes_required
        candidate_node_groups = self.get_candidate_node_groups(job, island_groups)
        best_subset = None
        best_score = float("-inf")

        for group in candidate_node_groups:
            if len(group) < n:
                continue

            # Empty nodes of the same family are identical, therefore only possibly need to evaluate n of them.   
            active = sorted(
                [nd for nd in group if nd.CPUs_in_use > 0 or nd.GPUs_in_use > 0 or nd.memory_in_use > 0],
                key=lambda node: (node.list_position, node.id)
            )
            empty = sorted(
                [nd for nd in group if nd.CPUs_in_use == 0 and nd.GPUs_in_use == 0 and nd.memory_in_use == 0],
                key=lambda node: (node.list_position, node.id)
            )
            candidates = active + empty[:n]
            candidates = self._shrink_candidates_to_threshold(candidates, job, n)

            if len(candidates) < n:
                continue

            for subset in combinations(candidates, n):
                if not self.nodes_have_capacity(subset, job):
                    continue
                subset_score = self._evaluate_subset(list(subset), job)
                if subset_score > best_score:
                    best_score = subset_score
                    best_subset = list(subset)

        return best_subset

class CopyRealNodeSelection(NodeSelectionStrategy): 
    """ Places jobs exactly where the real logs placed them for benchmarking. """ 
    def __init__(self): 
        self.map_names_to_nodes = {} 
        
    
    def select_nodes(self, job, island_groups):
        if not self.map_names_to_nodes:
            self.map_names_to_nodes = {
                node.name: node
                for groups in island_groups.values()
                for group in groups
                for node in group
            }

        nodes = []
        if not job.real_node_selection:
            return nodes, {"failed": True, "reason": "no_real_node_selection_documented"}
        for node in job.real_node_selection:
            nodes.append(self.map_names_to_nodes[node])
        return nodes, {"failed": False, "reason": ""}


class NaiveFeasible(NodeSelectionStrategy):
    """
    Fixed-order feasible baseline.

    Uses static cluster order only. If the full combinatorial search is too large,
    later nodes are trimmed until the subset count falls below the threshold.
    """

    def score_subset(self, nodes, job):
        return 0.0

    def score_single_node(self, node, job):
        return -float(node.list_position)

    def select_nodes(self, job, island_groups):
        if job.nodes_required <= 0:
            return [], {"failed": True, "reason": "invalid_nodes_required"}

        n = job.nodes_required
        candidate_node_groups = self.get_candidate_node_groups(job, island_groups)

        for group in candidate_node_groups:
            if len(group) < n:
                continue

            candidates = sorted(group, key=lambda node: (node.list_position, node.id))

            while len(candidates) > n and self._count_subsets(len(candidates), n) > self.MAX_FULL_SUBSET_EVALUATIONS:
                candidates.pop()

            for subset in combinations(candidates, n):
                if self.nodes_have_capacity(subset, job):
                    return list(subset), {"failed": False, "reason": "first_fixed_order_feasible"}

        return [], {"failed": True, "reason": "no_feasible_subset_found"}
    


class Active_Node_Reuse(NodeSelectionStrategy):
    """
    Prefer feasible subsets that reuse already-active nodes.
    """
    def _is_active(self, node):
        return (
            node.CPUs_in_use > 0
            or node.GPUs_in_use > 0
            or node.memory_in_use > 0
        )

    def score_subset(self, nodes, job):
        return sum(1.0 for node in nodes if self._is_active(node))

    def score_single_node(self, node, job):
        return 1.0 if self._is_active(node) else 0.0

    def _evaluate_subset(self, nodes, job):
        # Snapshot activity state BEFORE the trial allocation.
        pre_score = sum(1.0 for node in nodes if self._is_active(node))

        # Still worth checking for errors here if resource distribution is impossible
        record = self.resource_distribution_strategy.allocate_resources(job, nodes)
        try:
            return pre_score
        finally:
            for node, (cpus, gpus, mem) in record.items():
                node.release_job(cpus, gpus, mem)

class LoadSpreading(NodeSelectionStrategy):
    """
    Prefer subsets that leave the most free capacity after filling, spreading load evenly.
    """
    def score_subset(self, nodes, job):
        total = 0.0
        for node in nodes:
            cpu_term = ((node.total_CPUs - node.CPUs_in_use) / node.total_CPUs) if node.total_CPUs > 0 else 0.0
            gpu_term = ((node.total_GPUs - node.GPUs_in_use) / node.total_GPUs) if node.total_GPUs > 0 else 0.0
            mem_term = ((node.total_memory - node.memory_in_use) / node.total_memory) if node.total_memory > 0 else 0.0
            total += cpu_term + gpu_term + mem_term
        return total
    
    def score_single_node(self, node, job) -> float:
        free_c = node.total_CPUs - node.CPUs_in_use
        free_g = node.total_GPUs - node.GPUs_in_use
        free_m = node.total_memory - node.memory_in_use

        cpu_term = (free_c / node.total_CPUs) if node.total_CPUs > 0 else 0.0
        gpu_term = (free_g / node.total_GPUs) if node.total_GPUs > 0 else 0.0
        mem_term = (free_m / node.total_memory) if node.total_memory > 0 else 0.0
        return cpu_term + gpu_term + mem_term


class CPU_Best_Fit(NodeSelectionStrategy):
    """
    Prefer subsets that leave the least CPU slack after filling.
    """
    def score_subset(self, nodes, job):
        return sum(
            -((node.total_CPUs - node.CPUs_in_use) / node.total_CPUs) if node.total_CPUs > 0 else 0.0
            for node in nodes
        )
    
    def score_single_node(self, node, job) -> float:
        n = max(job.nodes_required, 1)
        share_c = job.CPUs_required / n
        free_c = node.total_CPUs - node.CPUs_in_use
        return -((free_c - share_c) / node.total_CPUs) if node.total_CPUs > 0 else 0.0


class GPU_Best_Fit(NodeSelectionStrategy):
    """
    Prefer subsets that leave the least GPU slack after filling.
    """
    def score_subset(self, nodes, job):
        return sum(
            -((node.total_GPUs - node.GPUs_in_use) / node.total_GPUs) if node.total_GPUs > 0 else 0.0
            for node in nodes
        )
    
    def score_single_node(self, node, job) -> float:
        n = max(job.nodes_required, 1)
        share_g = job.GPUs_required / n
        free_g = node.total_GPUs - node.GPUs_in_use
        return -((free_g - share_g) / node.total_GPUs) if node.total_GPUs > 0 else 0.0


class Node_Dependent_Best_Fit(NodeSelectionStrategy):
    """CPU best fit on ruby nodes, GPU best fit on all other node families."""

    _cpu = CPU_Best_Fit()
    _gpu = GPU_Best_Fit()

    def score_subset(self, nodes, job):
        if nodes[0].node_type == "ruby":
            return self._cpu.score_subset(nodes, job)
        return self._gpu.score_subset(nodes, job)

    def score_single_node(self, node, job) -> float:
        if node.node_type == "ruby":
            return self._cpu.score_single_node(node, job)
        return self._gpu.score_single_node(node, job)


class Manhattan_Slack_Best_Fit(NodeSelectionStrategy):
    """
    Prefer subsets that minimise total normalised remaining slack across all resource dimensions.
    """
    def score_subset(self, nodes, job):
        total = 0.0
        for node in nodes:
            cpu_term = ((node.total_CPUs - node.CPUs_in_use) / node.total_CPUs) if node.total_CPUs > 0 else 0.0
            gpu_term = ((node.total_GPUs - node.GPUs_in_use) / node.total_GPUs) if node.total_GPUs > 0 else 0.0
            mem_term = ((node.total_memory - node.memory_in_use) / node.total_memory) if node.total_memory > 0 else 0.0
            total += cpu_term + gpu_term + mem_term
        return -total

    def score_single_node(self, node, job) -> float:
        n = max(job.nodes_required, 1)

        share_c = job.CPUs_required / n
        share_g = job.GPUs_required / n
        share_m = job.memory_required / n

        free_c = node.total_CPUs - node.CPUs_in_use
        free_g = node.total_GPUs - node.GPUs_in_use
        free_m = node.total_memory - node.memory_in_use

        cpu_term = ((free_c - share_c) / node.total_CPUs) if node.total_CPUs > 0 else 0.0
        gpu_term = ((free_g - share_g) / node.total_GPUs) if node.total_GPUs > 0 else 0.0
        mem_term = ((free_m - share_m) / node.total_memory) if node.total_memory > 0 else 0.0

        return -(cpu_term + gpu_term + mem_term)

class Dominant_Resource_Best_Fit(NodeSelectionStrategy):
    """
    Identify the job's dominant resource dimension per node, then minimise remaining
    slack on that dimension after filling.
    """
    def score_subset(self, nodes, job):
        total = 0.0
        for node in nodes:
            cpu_share = (job.CPUs_required / node.total_CPUs) if node.total_CPUs > 0 else float("-inf")
            gpu_share = (job.GPUs_required / node.total_GPUs) if node.total_GPUs > 0 else float("-inf")
            mem_share = (job.memory_required / node.total_memory) if node.total_memory > 0 else float("-inf")

            dominant = max(
                (cpu_share, "cpu"), (gpu_share, "gpu"), (mem_share, "mem"),
                key=lambda x: x[0]
            )[1]

            if dominant == "cpu":
                total += -((node.total_CPUs - node.CPUs_in_use) / node.total_CPUs) if node.total_CPUs > 0 else float("-inf")
            elif dominant == "gpu":
                total += -((node.total_GPUs - node.GPUs_in_use) / node.total_GPUs) if node.total_GPUs > 0 else float("-inf")
            else:
                total += -((node.total_memory - node.memory_in_use) / node.total_memory) if node.total_memory > 0 else float("-inf")
        return total
    

    def score_single_node(self, node, job) -> float:
        n = max(job.nodes_required, 1)

        share_c = job.CPUs_required / n
        share_g = job.GPUs_required / n
        share_m = job.memory_required / n

        cpu_share = (share_c / node.total_CPUs) if node.total_CPUs > 0 else float("-inf")
        gpu_share = (share_g / node.total_GPUs) if node.total_GPUs > 0 else float("-inf")
        mem_share = (share_m / node.total_memory) if node.total_memory > 0 else float("-inf")

        dominant = max(
            (cpu_share, "cpu"),
            (gpu_share, "gpu"),
            (mem_share, "mem"),
            key=lambda x: x[0]
        )[1]

        free_c = node.total_CPUs - node.CPUs_in_use
        free_g = node.total_GPUs - node.GPUs_in_use
        free_m = node.total_memory - node.memory_in_use

        if dominant == "cpu":
            return -((free_c - share_c) / node.total_CPUs) if node.total_CPUs > 0 else float("-inf")
        elif dominant == "gpu":
            return -((free_g - share_g) / node.total_GPUs) if node.total_GPUs > 0 else float("-inf")
        else:
            return -((free_m - share_m) / node.total_memory) if node.total_memory > 0 else float("-inf")


class Workload_Aware_Weighted_Manhattan_Slack(NodeSelectionStrategy):
    """
    Weighted Manhattan slack minimiser.

    Score:  -sum_nodes[ w_c*free_c/total_c + w_g*free_g/total_g + w_m*free_m/total_m ]
    """
    def score_subset(self, nodes, job):
        total = 0.0
        for node in nodes:
            w_c, w_g, w_m = self._get_weights(node)
            cpu_term = w_c * ((node.total_CPUs - node.CPUs_in_use) / node.total_CPUs) if node.total_CPUs > 0 else 0.0
            gpu_term = w_g * ((node.total_GPUs - node.GPUs_in_use) / node.total_GPUs) if node.total_GPUs > 0 else 0.0
            mem_term = w_m * ((node.total_memory - node.memory_in_use) / node.total_memory) if node.total_memory > 0 else 0.0
            total += cpu_term + gpu_term + mem_term
        return -total
    
    
    def score_single_node(self, node, job) -> float:
        n = max(job.nodes_required, 1)

        share_c = job.CPUs_required / n
        share_g = job.GPUs_required / n
        share_m = job.memory_required / n

        free_c = node.total_CPUs - node.CPUs_in_use
        free_g = node.total_GPUs - node.GPUs_in_use
        free_m = node.total_memory - node.memory_in_use

        w_c, w_g, w_m = self._get_weights(node)

        cpu_term = w_c * ((free_c - share_c) / node.total_CPUs) if node.total_CPUs > 0 else 0.0
        gpu_term = w_g * ((free_g - share_g) / node.total_GPUs) if node.total_GPUs > 0 else 0.0
        mem_term = w_m * ((free_m - share_m) / node.total_memory) if node.total_memory > 0 else 0.0

        return -(cpu_term + gpu_term + mem_term)


class Workload_Aware_Weighted_Dominant_Resource(NodeSelectionStrategy):
    """
    Workload-aware dominant-resource best fit.

    Dominant dimension per node: argmax_i[ w_i * (job_i / total_i) ]
    Score: negative normalised remaining slack on that dimension after filling.
    """
    def score_subset(self, nodes, job):
        total = 0.0
        for node in nodes:
            w_c, w_g, w_m = self._get_weights(node)

            cpu_share = w_c * (job.CPUs_required / node.total_CPUs) if node.total_CPUs > 0 else float("-inf")
            gpu_share = w_g * (job.GPUs_required / node.total_GPUs) if node.total_GPUs > 0 else float("-inf")
            mem_share = w_m * (job.memory_required / node.total_memory) if node.total_memory > 0 else float("-inf")

            dominant = max(
                (cpu_share, "cpu"), (gpu_share, "gpu"), (mem_share, "mem"),
                key=lambda x: x[0]
            )[1]

            if dominant == "cpu":
                total += -((node.total_CPUs - node.CPUs_in_use) / node.total_CPUs) if node.total_CPUs > 0 else float("-inf")
            elif dominant == "gpu":
                total += -((node.total_GPUs - node.GPUs_in_use) / node.total_GPUs) if node.total_GPUs > 0 else float("-inf")
            else:
                total += -((node.total_memory - node.memory_in_use) / node.total_memory) if node.total_memory > 0 else float("-inf")
        return total
    
    
    def score_single_node(self, node, job) -> float:
        n = max(job.nodes_required, 1)

        share_c = job.CPUs_required / n
        share_g = job.GPUs_required / n
        share_m = job.memory_required / n

        w_c, w_g, w_m = self._get_weights(node)

        cpu_share = w_c * (share_c / node.total_CPUs) if node.total_CPUs > 0 else float("-inf")
        gpu_share = w_g * (share_g / node.total_GPUs) if node.total_GPUs > 0 else float("-inf")
        mem_share = w_m * (share_m / node.total_memory) if node.total_memory > 0 else float("-inf")

        dominant = max(
            (cpu_share, "cpu"),
            (gpu_share, "gpu"),
            (mem_share, "mem"),
            key=lambda x: x[0]
        )[1]

        free_c = node.total_CPUs - node.CPUs_in_use
        free_g = node.total_GPUs - node.GPUs_in_use
        free_m = node.total_memory - node.memory_in_use

        if dominant == "cpu":
            return -((free_c - share_c) / node.total_CPUs) if node.total_CPUs > 0 else float("-inf")
        elif dominant == "gpu":
            return -((free_g - share_g) / node.total_GPUs) if node.total_GPUs > 0 else float("-inf")
        else:
            return -((free_m - share_m) / node.total_memory) if node.total_memory > 0 else float("-inf")



class DeadlineAware_CoSchedule(NodeSelectionStrategy):
    """
    Base class for temporal co-scheduling heuristics.

    Tracks the latest expected finish time per node and prefers placements where
    a job's finish time aligns with (or falls before) the node's latest current
    finish, so nodes drain in sync.

    Subclasses override _get_job_deadline to pick the time source.
    """

    def __init__(self, family_weights=None):
        super().__init__(family_weights)
        self.node_deadlines = {}  # node -> {job_id: deadline_timestamp}

    # Called by subclass to choose deadline source
    def _get_job_deadline(self, job):
        raise NotImplementedError

    # Called by SlurmSimulator to update latest deadline on nodes after placement is confirmed
    def register_placement(self, job, nodes):
        deadline = self._get_job_deadline(job)
        if deadline is None:
            deadline = datetime.max  # unknown -> treat as infinitely late
        for node in nodes:
            self.node_deadlines.setdefault(node, {})[job.id] = deadline

    # Called by SlurmSimulator to update latest deadline on nodes after releasing a job    
    def register_release(self, job_id, nodes):
        for node in nodes:
            entries = self.node_deadlines.get(node)
            if entries is None:
                continue
            entries.pop(job_id, None)
            if not entries:
                del self.node_deadlines[node]

    # Find the latest deadline of jobs currently on node
    def _latest_deadline_on(self, node):
        entries = self.node_deadlines.get(node)
        if not entries:
            return None
        return max(entries.values())

    
    # Score the cost of placing a job on a given node
    def _node_cost_seconds(self, node, job):
        job_deadline = self._get_job_deadline(job)
        if job_deadline is None:
            # Placing a deadline-less job: fall back to preferring active over empty by a flat margin.
            return 0.0 if self.node_deadlines.get(node) else 86400.0

        latest = self._latest_deadline_on(node)
        if latest is None:
            # Empty node: activation cost = full job duration
            return max(0.0, (job_deadline - job.start_time).total_seconds())

        if job_deadline <= latest:
            return 0.0
        return (job_deadline - latest).total_seconds()

    def score_single_node(self, node, job):
        return -self._node_cost_seconds(node, job)

    def score_subset(self, nodes, job):
        return -sum(self._node_cost_seconds(n, job) for n in nodes)

    def _evaluate_subset(self, nodes, job):
        # Score from pre-allocation state (same pattern as Active_Node_Reuse).
        pre_score = self.score_subset(nodes, job)
        record = self.resource_distribution_strategy.allocate_resources(job, nodes)
        try:
            return pre_score
        finally:
            for node, (cpus, gpus, mem) in record.items():
                node.release_job(cpus, gpus, mem)


class Timelimit_CoSchedule(DeadlineAware_CoSchedule):
    """
    Realistic temporal heuristic: uses user-requested walltime (estimated_deadline).
    """
    def _get_job_deadline(self, job):
        return job.estimated_deadline


class Perfect_CoSchedule(DeadlineAware_CoSchedule):
    """
    Upper-bound temporal heuristic: uses actual end time from the logs.
    Not available to real schedulers; used to bound the value of temporal info.
    """
    def _get_job_deadline(self, job):
        return job.end_time



class ResourceDistributionStrategy():
    """Base class for resource distribution when a job is assigned to multiple nodes"""
    def allocate_resources(self, job, nodes):
        raise NotImplementedError


class GreedyBlockFill(ResourceDistributionStrategy):
    """
    Mimics Iridis X's resource distribution logic
    Default SLURM behaviour as confirmed in our config file is that when multiple nodes are allocated, SLURM will pack all the resources onto a singular node, then move onto the next in a greedy fashion.
    For example, if we have 3 nodes, and need to allocate 100CPUs across them, it will fill node 1 and 2, then partially fill 3.
    It chooses the order of filling nodes based on a the order they appear in slurm.conf, for example - node001 filled first, then node002, then node003.
    There is no consideration for which node to fill first by default, just uses the list of names as in slurm.conf
    """
    def allocate_resources(self, job, nodes):
        sorted_nodes = sorted(nodes, key=lambda node: (node.list_position, node.id))

        CPUs_required = job.CPUs_required
        GPUs_required = job.GPUs_required
        memory_required = job.memory_required

        resource_distribution_record = {}

        for node in sorted_nodes:
            CPU_allocation = 0
            GPU_allocation = 0
            memory_allocation = 0

            if CPUs_required == 0 and GPUs_required == 0 and memory_required == 0:
                break

            if CPUs_required > 0:
                free_CPUs = node.total_CPUs - node.CPUs_in_use
                CPU_allocation = min(CPUs_required, free_CPUs)
                CPUs_required -= CPU_allocation

            if GPUs_required > 0:
                free_GPUs = node.total_GPUs - node.GPUs_in_use
                GPU_allocation = min(GPUs_required, free_GPUs)
                GPUs_required -= GPU_allocation

            if memory_required > 0:
                free_memory = node.total_memory - node.memory_in_use
                memory_allocation = min(memory_required, free_memory)
                memory_required -= memory_allocation

            resource_distribution_record[node] = (CPU_allocation, GPU_allocation, memory_allocation)
            node.run_job(CPU_allocation, GPU_allocation, memory_allocation)

        if CPUs_required != 0 or GPUs_required != 0 or memory_required != 0:
            for node, (cpus, gpus, mem) in resource_distribution_record.items():
                node.release_job(cpus, gpus, mem)
            raise ValueError(
                f"Incomplete allocation for job {job.id}: "
                f"remaining CPUs={CPUs_required}, GPUs={GPUs_required}, memory={memory_required}"
            )

        return resource_distribution_record


class EvenSplit(ResourceDistributionStrategy):
    """
    Capacity-aware approximate even split.

    For each resource:
    1. Repeatedly divide the remaining demand evenly across nodes that still have spare capacity.
    2. Cap each node by its current free capacity.
    3. Continue until all demand is allocated or fail if impossible.
    """

    def _split_evenly(self, total_required, free_caps):
        n = len(free_caps)
        allocs = [0] * n
        remaining = total_required

        while remaining > 0:
            available = [i for i in range(n) if allocs[i] < free_caps[i]]
            if not available:
                raise ValueError(f"Could not allocate remaining amount: {remaining}")

            share = math.ceil(remaining / len(available))

            allocated_this_round = 0
            for i in available:
                room = free_caps[i] - allocs[i]
                give = min(share, room, remaining)
                allocs[i] += give
                remaining -= give
                allocated_this_round += give

                if remaining == 0:
                    break

            if allocated_this_round == 0:
                raise ValueError(f"Allocation stalled with {remaining} remaining")

        return allocs

    def allocate_resources(self, job, nodes):
        sorted_nodes = sorted(nodes, key=lambda node: (node.list_position, node.id))

        free_cpus = [node.total_CPUs - node.CPUs_in_use for node in sorted_nodes]
        free_gpus = [node.total_GPUs - node.GPUs_in_use for node in sorted_nodes]
        free_mem = [node.total_memory - node.memory_in_use for node in sorted_nodes]

        if sum(free_cpus) < job.CPUs_required:
            raise ValueError(f"Not enough CPU capacity for job {job.id}")
        if sum(free_gpus) < job.GPUs_required:
            raise ValueError(f"Not enough GPU capacity for job {job.id}")
        if sum(free_mem) < job.memory_required:
            raise ValueError(f"Not enough memory capacity for job {job.id}")

        cpu_allocs = self._split_evenly(job.CPUs_required, free_cpus)
        gpu_allocs = self._split_evenly(job.GPUs_required, free_gpus)
        mem_allocs = self._split_evenly(job.memory_required, free_mem)

        resource_distribution_record = {}

        try:
            for node, cpu_a, gpu_a, mem_a in zip(sorted_nodes, cpu_allocs, gpu_allocs, mem_allocs):
                resource_distribution_record[node] = (cpu_a, gpu_a, mem_a)
                node.run_job(cpu_a, gpu_a, mem_a)
        except Exception:
            for node, (cpu_a, gpu_a, mem_a) in resource_distribution_record.items():
                node.release_job(cpu_a, gpu_a, mem_a)
            raise

        return resource_distribution_record
        


class SlurmSimulation:
    def __init__(self, island_groups, node_selection_strategy, resource_distribution_strategy):
        self.island_groups = island_groups
        self.node_selection_strategy = node_selection_strategy
        self.resource_distribution_strategy = resource_distribution_strategy
        self.node_selection_strategy.resource_distribution_strategy = resource_distribution_strategy
        self.job_tracker = {}

    def place_job(self, job):
        selected_nodes = []
        info = {}
        try:
            selected_nodes, info = self.node_selection_strategy.select_nodes(job, self.island_groups)

            if not selected_nodes:
                return False, info

            resource_distribution_record = self.resource_distribution_strategy.allocate_resources(job, selected_nodes)
            self.job_tracker[job.id] = resource_distribution_record

            if hasattr(self.node_selection_strategy, "register_placement"):
                self.node_selection_strategy.register_placement(job, selected_nodes)

            return True, info

        except ValueError:
            return False, {
                "failed": True,
                "reason": "resource_distribution_failed",
            }


    def release_job(self, id):
        resource_distribution_record = self.job_tracker.pop(id, None)
        if resource_distribution_record is None:
            return False

        nodes = list(resource_distribution_record.keys())
        for node, (cpus, gpus, mem) in resource_distribution_record.items():
            node.release_job(cpus, gpus, mem)

        if hasattr(self.node_selection_strategy, "register_release"):
            self.node_selection_strategy.register_release(id, nodes)

        return True


    def get_current_state(self):
        """Return current cluster state for external logging"""
        all_nodes = [node for groups in self.island_groups.values() for group in groups for node in group]
        return {
            'active_jobs': len(self.job_tracker),
            'nodes': [{
                'name': n.name,
                'physical_CPUs': n.physical_CPUs,
                'cores_per_physical_CPU': n.cores_per_physical_CPU,
                'CPUs_in_use': n.CPUs_in_use,
                'GPUs_in_use': n.GPUs_in_use,
                'memory_in_use': n.memory_in_use,
                'total_CPUs' : n.total_CPUs,
                'total_GPUs' : n.total_GPUs,
                'total_memory' : n.total_memory,
                'power_consumption' : n.current_power_consumption
                }
            for n in all_nodes]
        }

   



        
