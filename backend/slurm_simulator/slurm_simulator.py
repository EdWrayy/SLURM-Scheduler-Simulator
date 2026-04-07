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
    def select_nodes(self, job, node_list, island_groups):
        """
        Returns: (selected_nodes, failure_info)
          - selected_nodes: list[Node] (empty if failure)
          - failure_info: dict with keys:
              {
                "failed": bool,
                "reason": str
              }
        """
        raise NotImplementedError

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
            "reason": "no_sufficient_node_set_found",
        }

    
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
            return nodes, {"failed": True, "reason": "no_real_node_selection_documented"}
        for node in job.real_node_selection:
            nodes.append(self.map_names_to_nodes[node])
        return nodes, {"failed": False, "reason": ""}


class NaiveFirstFit(NodeSelectionStrategy):
    """
    Slide a window of n nodes and return the first set of n nodes found which meets criteria
    This will not find non-contigous subsets 
    """

    def select_nodes(self, job, node_list, island_groups):
        n = job.nodes_required

        if n <= 0:
            return [], {
                "failed": True,
                "reason": "invalid_nodes_required"
            }

        candidate_node_groups = []
        for allowed_type in job.allowed_node_types:
            candidate_node_groups.extend(island_groups[allowed_type])

        for group in candidate_node_groups:
            eligible = sorted(group, key=lambda nd: nd.list_position)
            for start in range(len(eligible) - n + 1):
                window = eligible[start:start + n]
                if self.nodes_have_capacity(window, job):
                    return window, {"failed": False, "reason": "first_fit"}

        return [], self.build_failure_info()


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
    def __init__(self, node_list, island_groups, node_selection_strategy, resource_distribution_strategy):
        self.node_list = node_list
        self.island_groups = island_groups
        self.node_selection_strategy = node_selection_strategy
        self.resource_distribution_strategy = resource_distribution_strategy
        self.job_tracker = {}

    def place_job(self, job):
        try:
            selected_nodes, info = self.node_selection_strategy.select_nodes(job, self.node_list, self.island_groups)

            if not selected_nodes:
                return False, info

            resource_distribution_record = self.resource_distribution_strategy.allocate_resources(job, selected_nodes)
            self.job_tracker[job.id] = resource_distribution_record

            return True, info

        except Exception as e:
            return False, {
                "failed": True,
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

   



        
