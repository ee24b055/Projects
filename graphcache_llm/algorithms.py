"""
Algorithmic Routing Layer: Topological Sort and Cache Propagation Logic.
"""
from typing import List, Set
from structures import ContextDAG, LRUCache

class RoutingEngine:
    @staticmethod
    def compute_topological_execution_order(dag: ContextDAG) -> List[str]:
        """
        Kahn's Algorithm for Topological Sort: Calculates the optimal, non-conflicting
        execution order of multi-agent tasks. Complexity: O(V + E)
        """
        in_deg = dag.in_degree.copy()
        queue = [node for node, deg in in_deg.items() if deg == 0]
        order = []
        
        while queue:
            curr = queue.pop(0)
            order.append(curr)
            
            for neighbor in dag.adjacency_list.get(curr, []):
                in_deg[neighbor] -= 1
                if in_deg[neighbor] == 0:
                    queue.append(neighbor)
                    
        if len(order) != len(dag.adjacency_list):
            raise ValueError("Cyclic dependency detected! Context loop broken.")
        return order

    @staticmethod
    def propagate_cache_invalidation(dag: ContextDAG, cache: LRUCache, stale_node: str) -> List[str]:
        """
        Graph Traversal (BFS): Triggers cascading downstream cache invalidation
        when a parent agent yields updated execution metrics.
        """
        invalidated_nodes = []
        queue = [stale_node]
        visited: Set[str] = set()
        
        while queue:
            curr = queue.pop(0)
            if curr not in visited:
                visited.add(curr)
                if cache.get(curr) is not None:
                    # Invalidate in cache
                    cache.put(curr, None) 
                    invalidated_nodes.append(curr)
                
                for neighbor in dag.adjacency_list.get(curr, []):
                    queue.append(neighbor)
                    
        return invalidated_nodes