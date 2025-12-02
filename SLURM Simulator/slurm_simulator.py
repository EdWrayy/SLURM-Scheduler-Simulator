class Node:
    def __init__(self, name, id, list_position, total_CPUs, total_GPUs, total_memory, GPU_type=None):
        self.name = name
        self.id = id 
        self.list_position = list_position #The position in the slurm.conf listing, relevant for resource distirbution.
        self.total_CPUs = total_CPUs
        self.total_GPUs = total_GPUs
        self.total_memory = total_memory
        self.CPUs_in_use = 0
        self.GPUs_in_use = 0
        self.memory_in_use = 0
        self.GPU_type = GPU_type


    def run_job(self, CPUs_required, GPUs_required, memory_required):
        self.CPUs_in_use += CPUs_required
        self.GPUs_in_use += GPUs_required
        self.memory_in_use += memory_required

    def release_job(self, CPUs_required, GPUs_required, memory_required):
        self.CPUs_in_use -= CPUs_required
        self.GPUs_in_use -= GPUs_required
        self.memory_in_use -= memory_required
        



class Job:
    def __init__(self, id, nodes_required, CPUs_required, GPUs_required, memory_required, start_time, end_time, real_node_selection=None) :
        self.id = id 
        self.nodes_required = nodes_required
        self.CPUs_required = CPUs_required
        self.GPUs_required = GPUs_required
        self.memory_required = memory_required
        self.start_time = start_time
        self.end_time = end_time
        self.node_utilisation = {}
        self.real_node_selection = real_node_selection #Exclusively for copy real node selection simulation, can be ignored otherwise


class NodeSelectionStrategy:
    """Base class for node selection strategies when placing jobs"""
    def select_nodes(self, job, available_nodes):
        """Returns list of nodes to place the job on"""
        raise NotImplementedError
    
    def has_capacity(self, node, job):
        return (node.total_CPUs - node.CPUs_in_use >= job.CPUs_required and
                node.total_GPUs - node.GPUs_in_use >= job.GPUs_required and
                node.total_memory - node.memory_in_use >= job.memory_required)


class CopyRealNodeSelection(NodeSelectionStrategy):
    """
    Places jobs exactly where the real logs placed them for benchmarking.
    """
    def __init__(self):
        self.map_names_to_nodes = {}

    def select_nodes(self, job, node_list):
        if not self.map_names_to_nodes:
            for node in node_list:
                self.map_names_to_nodes[node.name] = node

        nodes = []
        for node in job.real_node_selection:
            nodes.append(self.map_names_to_nodes[node])
        return nodes
        

class FirstFitNodeSelection(NodeSelectionStrategy):
    """Place job on first available nodes that fit"""
    def select_nodes(self, job, node_list):
        selected = []
        node_index = 0
        while len(selected) < job.nodes_required and node_index < len(node_list):
            candidate_node = node_list[node_index]
            if self.has_capacity(candidate_node, job):
                selected.append(candidate_node)
            node_index += 1
        
        return [] if len(selected) < job.nodes_required else selected


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
    def __init__(self, cluster_name, node_list, node_selection_strategy, resource_distribution_strategy):
        self.cluster_name = cluster_name
        self.node_list = node_list
        self.node_selection_strategy = node_selection_strategy
        self.resource_distribution_strategy = resource_distribution_strategy
        self.job_tracker = {}


    def place_job(self, job):
        selected_nodes = self.node_selection_strategy.select_nodes(job, self.node_list)
        if not selected_nodes :
            raise Exception("Insufficient Capacity to Place Job!")
        resource_distribution_record = self.resource_distribution_strategy.allocate_resources(job, selected_nodes)
        self.job_tracker[job.id] = resource_distribution_record

    
    def release_job(self, id):
        resource_distribution_record = self.job_tracker.pop(id, None)
        if resource_distribution_record is None:
            print("Tried to release a non-existent job with id: " + id)
            return
        
        for node,(cpus, gpus, mem) in resource_distribution_record.items():
            node.release_job(cpus, gpus, mem)

    def get_current_state(self):
        """Return current cluster state for external logging"""
        return {
            'active_jobs': len(self.job_tracker),
            'nodes': [{
                'name': n.name,
                'CPUs_in_use': n.CPUs_in_use,
                'GPUs_in_use': n.GPUs_in_use,
                'memory_in_use': n.memory_in_use,
                'total_CPUs' : n.total_CPUs,
                'total_GPUs' : n.total_GPUs,
                'total_memory' : n.total_memory
                } 
            for n in self.node_list]
        }

   



        


