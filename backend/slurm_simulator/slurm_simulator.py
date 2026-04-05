from backend.common.models import Job
import math


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
    def select_nodes(self, job, node_list):
        """
        Returns: (selected_nodes, failure_info)
          - selected_nodes: list[Node] (empty if failure)
          - failure_info: dict with keys like:
              {
                "failed": bool,
                "limiting_resources": ["CPUs", "GPUs", "memory"],
                "reason": "insufficient_total_capacity" | "other"
              }
        """
        raise NotImplementedError

    def free_capacity(self, node):
        return (
            node.total_CPUs - node.CPUs_in_use,
            node.total_GPUs - node.GPUs_in_use,
            node.total_memory - node.memory_in_use,
        )

    def has_capacity_for_all(self, node, job):
        free_c, free_g, free_m = self.free_capacity(node)
        return (
            free_c >= job.CPUs_required and
            free_g >= job.GPUs_required and
            free_m >= job.memory_required
        )

    def build_failure_info(self, job, node_list):
        total_free_cpus = 0
        total_free_gpus = 0
        total_free_mem = 0

        for node in node_list:
            free_c, free_g, free_m = self.free_capacity(node)
            total_free_cpus += free_c
            total_free_gpus += free_g
            total_free_mem += free_m

        limiting = []
        if total_free_cpus < job.CPUs_required:
            limiting.append("CPUs")
        if total_free_gpus < job.GPUs_required:
            limiting.append("GPUs")
        if total_free_mem < job.memory_required:
            limiting.append("memory")

        if not limiting:
            limiting = ["fragmentation_or_constraints"]

        return {
            "failed": True,
            "limiting_resources": limiting,
            "reason": "insufficient_total_capacity" if limiting != ["fragmentation_or_constraints"] else "fragmentation_or_constraints",
        }

    def spread_select(self, job, ordered_nodes, success_reason):
        """
        Spreads resource selection across nodes when we cannot fit an entire job on any one node
        """
        cpus_left = job.CPUs_required
        gpus_left = job.GPUs_required
        mem_left = job.memory_required

        selected = []

        for node in ordered_nodes:
            if cpus_left <= 0 and gpus_left <= 0 and mem_left <= 0:
                break

            free_c, free_g, free_m = self.free_capacity(node)
            can_help = (
                (cpus_left > 0 and free_c > 0) or
                (gpus_left > 0 and free_g > 0) or
                (mem_left > 0 and free_m > 0)
            )

            if not can_help:
                continue

            selected.append(node)
            cpus_left = max(0, cpus_left - free_c)
            gpus_left = max(0, gpus_left - free_g)
            mem_left = max(0, mem_left - free_m)

        if cpus_left <= 0 and gpus_left <= 0 and mem_left <= 0:
            return selected, {
                "failed": False,
                "limiting_resources": [],
                "reason": success_reason,
            }

        return [], self.build_failure_info(job, ordered_nodes)
    
    def filter_by_type(self, job, node_list):
        if not job.allowed_node_types:
            return node_list
        return [n for n in node_list if n.node_type in job.allowed_node_types]

class CopyRealNodeSelection(NodeSelectionStrategy): 
    """ Places jobs exactly where the real logs placed them for benchmarking. """ 
    def __init__(self): 
        self.map_names_to_nodes = {} 
        
    
    def select_nodes(self, job, node_list): 
        if not self.map_names_to_nodes: 
            for node in node_list: 
                self.map_names_to_nodes[node.name] = node 
        
        nodes = [] 
        if not job.real_node_selection: 
            return nodes, {"failed": True, "limiting_resources": [], "reason": "No Real Node Selection Documented"}
        for node in job.real_node_selection: 
            nodes.append(self.map_names_to_nodes[node]) 
        return nodes, {"failed": False, "limiting_resources": [], "reason": ""}


class FirstFitNodeSelection(NodeSelectionStrategy):
    """
    First-fit with single-node preference:
      1) If any single node can satisfy all requested resources, pick the first such node.
      2) Otherwise, spread across nodes in first-fit order until requirements are met.
      3) If still not possible, return failure and limiting resources.
    """

    def select_nodes(self, job, node_list):
        # 1) Prefer a single node that can host everything
        node_list = self.filter_by_type(job, node_list)
        for node in node_list:
            if self.has_capacity_for_all(node, job):
                return [node], {
                    "failed": False,
                    "limiting_resources": [],
                    "reason": "single_node_fit"
                }

        # 2) Otherwise, spread across nodes in order
        cpus_left = job.CPUs_required
        gpus_left = job.GPUs_required
        mem_left  = job.memory_required

        selected = []

        for node in node_list:
            if cpus_left <= 0 and gpus_left <= 0 and mem_left <= 0:
                break

            free_c, free_g, free_m = self.free_capacity(node)

            can_help = (cpus_left > 0 and free_c > 0) or (gpus_left > 0 and free_g > 0) or (mem_left > 0 and free_m > 0)
            if not can_help:
                continue

            selected.append(node)

            cpus_left = max(0, cpus_left - free_c)
            gpus_left = max(0, gpus_left - free_g)
            mem_left  = max(0, mem_left  - free_m)

        if cpus_left <= 0 and gpus_left <= 0 and mem_left <= 0:
            return selected, {
                "failed": False,
                "limiting_resources": [],
                "reason": "spread_fit"
            }

        # 3) Failure: work out what limited placement
        return [], self.build_failure_info(job, node_list)


class BestFitByFreeCPUsNodeSelection(NodeSelectionStrategy):
    """
    CPU-oriented best-fit:
      - Single node: minimise remaining CPUs after placement.
      - Multi-node: greedily consume nodes with the least free CPUs first.
    """

    def select_nodes(self, job, node_list):
        node_list = self.filter_by_type(job, node_list)
        candidates = [node for node in node_list if self.has_capacity_for_all(node, job)]
        if candidates:
            best = min(
                candidates,
                key=lambda node: (
                    (node.total_CPUs - node.CPUs_in_use) - job.CPUs_required,
                    (node.total_GPUs - node.GPUs_in_use) - job.GPUs_required,
                    (node.total_memory - node.memory_in_use) - job.memory_required,
                    node.list_position,
                    node.id,
                ),
            )
            return [best], {
                "failed": False,
                "limiting_resources": [],
                "reason": "single_node_best_fit_cpu",
            }

        ordered = sorted(
            node_list,
            key=lambda node: (
                node.total_CPUs - node.CPUs_in_use,
                node.list_position,
                node.id,
            ),
        )
        return self.spread_select(job, ordered, "spread_best_fit_cpu")


class BestFitByFreeGPUsNodeSelection(NodeSelectionStrategy):
    """
    GPU-oriented best-fit:
      - Single node: minimise remaining GPUs after placement.
      - Multi-node: greedily consume nodes with the least free GPUs first.
    """

    def select_nodes(self, job, node_list):
        node_list = self.filter_by_type(job, node_list)
        candidates = [node for node in node_list if self.has_capacity_for_all(node, job)]
        if candidates:
            best = min(
                candidates,
                key=lambda node: (
                    (node.total_GPUs - node.GPUs_in_use) - job.GPUs_required,
                    (node.total_CPUs - node.CPUs_in_use) - job.CPUs_required,
                    (node.total_memory - node.memory_in_use) - job.memory_required,
                    node.list_position,
                    node.id,
                ),
            )
            return [best], {
                "failed": False,
                "limiting_resources": [],
                "reason": "single_node_best_fit_gpu",
            }

        ordered = sorted(
            node_list,
            key=lambda node: (
                node.total_GPUs - node.GPUs_in_use,
                node.list_position,
                node.id,
            ),
        )
        return self.spread_select(job, ordered, "spread_best_fit_gpu")


class JointCpuGpuBestFitNodeSelection(NodeSelectionStrategy):
    """
    Joint CPU/GPU best-fit.
    Minimises Manhattan slack: remaining_CPU + remaining_GPU.
    """

    def select_nodes(self, job, node_list):
        node_list = self.filter_by_type(job, node_list)
        candidates = [node for node in node_list if self.has_capacity_for_all(node, job)]
        if candidates:
            best = min(
                candidates,
                key=lambda node: (
                    ((node.total_CPUs - node.CPUs_in_use) - job.CPUs_required) +
                    ((node.total_GPUs - node.GPUs_in_use) - job.GPUs_required),
                    (node.total_memory - node.memory_in_use) - job.memory_required,
                    node.list_position,
                    node.id,
                ),
            )
            return [best], {
                "failed": False,
                "limiting_resources": [],
                "reason": "single_node_joint_cpu_gpu_best_fit",
            }

        ordered = sorted(
            node_list,
            key=lambda node: (
                (node.total_CPUs - node.CPUs_in_use) + (node.total_GPUs - node.GPUs_in_use),
                node.list_position,
                node.id,
            ),
        )
        return self.spread_select(job, ordered, "spread_joint_cpu_gpu_best_fit")


class DominantResourcePackingNodeSelection(NodeSelectionStrategy):
    """
    Place jobs towards nodes already most loaded in the job's dominant resource.
    Dominant resource is chosen between CPU and GPU by higher requested proportion
    of currently available cluster capacity.
    """

    def _dominant_resource(self, job, node_list):
        total_free_cpus = sum(max(0, n.total_CPUs - n.CPUs_in_use) for n in node_list)
        total_free_gpus = sum(max(0, n.total_GPUs - n.GPUs_in_use) for n in node_list)

        cpu_pressure = (
            (job.CPUs_required / total_free_cpus)
            if total_free_cpus > 0 else float("inf") if job.CPUs_required > 0 else 0.0
        )
        gpu_pressure = (
            (job.GPUs_required / total_free_gpus)
            if total_free_gpus > 0 else float("inf") if job.GPUs_required > 0 else 0.0
        )

        return "GPUs" if gpu_pressure > cpu_pressure else "CPUs"

    def _dominant_utilisation(self, node, dominant):
        if dominant == "GPUs":
            return (node.GPUs_in_use / node.total_GPUs) if node.total_GPUs > 0 else 0.0
        return (node.CPUs_in_use / node.total_CPUs) if node.total_CPUs > 0 else 0.0

    def select_nodes(self, job, node_list):
        node_list = self.filter_by_type(job, node_list)
        dominant = self._dominant_resource(job, node_list)

        candidates = [node for node in node_list if self.has_capacity_for_all(node, job)]
        if candidates:
            best = min(
                candidates,
                key=lambda node: (
                    -self._dominant_utilisation(node, dominant),
                    ((node.total_GPUs - node.GPUs_in_use) - job.GPUs_required) if dominant == "GPUs"
                    else ((node.total_CPUs - node.CPUs_in_use) - job.CPUs_required),
                    (node.total_memory - node.memory_in_use) - job.memory_required,
                    node.list_position,
                    node.id,
                ),
            )
            return [best], {
                "failed": False,
                "limiting_resources": [],
                "reason": f"single_node_dominant_resource_packing_{dominant.lower()}",
            }

        ordered = sorted(
            node_list,
            key=lambda node: (
                -self._dominant_utilisation(node, dominant),
                (node.total_GPUs - node.GPUs_in_use) if dominant == "GPUs"
                else (node.total_CPUs - node.CPUs_in_use),
                node.list_position,
                node.id,
            ),
        )
        return self.spread_select(job, ordered, f"spread_dominant_resource_packing_{dominant.lower()}")



class ResourceDistributionStrategy():
    """Base class for resource distribution when a job is assigned to multiple nodes"""
    def allocate_resources(self, job, nodes):
        raise NotImplementedError


class DefaultResourceDistribution(ResourceDistributionStrategy):
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

            resource_distribution_record[node] = CPU_allocation, GPU_allocation, memory_allocation
            node.run_job(CPU_allocation, GPU_allocation, memory_allocation)

        return resource_distribution_record




        



class SlurmSimulation:
    def __init__(self, node_list, node_selection_strategy, resource_distribution_strategy):
        self.node_list = node_list
        self.node_selection_strategy = node_selection_strategy
        self.resource_distribution_strategy = resource_distribution_strategy
        self.job_tracker = {}

    def place_job(self, job):
        try:
            selected_nodes, info = self.node_selection_strategy.select_nodes(job, self.node_list)

            if not selected_nodes:
                return False, info

            resource_distribution_record = self.resource_distribution_strategy.allocate_resources(job, selected_nodes)
            self.job_tracker[job.id] = resource_distribution_record

            return True, info

        except Exception as e:
            return False, {
                "failed": True,
                "limiting_resources": f"{type(e).__name__}: {e}",
                "reason": f"{type(e).__name__}: {e}",
            }


    def release_job(self, id):
        resource_distribution_record = self.job_tracker.pop(id, None)
        if resource_distribution_record is None:
            return False

        for node, (cpus, gpus, mem) in resource_distribution_record.items():
            node.release_job(cpus, gpus, mem)
        return True


    def get_current_state(self):
        """Return current cluster state for external logging"""
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
            for n in self.node_list]
        }

   



        
