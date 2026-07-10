"""
Core Data Structures: Custom Doubly-Linked List, LRU Cache, and Directed Acyclic Graph (DAG).
Written from scratch to prove foundational CS mastery.
"""
from typing import Dict, Any, Optional, List

class Node:
    """A Node for the Doubly-Linked List used in the LRU Cache."""
    def __init__(self, key: str, val: Any):
        self.key: str = key
        self.val: Any = val
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None

class LRUCache:
    """Custom O(1) LRU Cache for managing memory bounds of context payloads."""
    def __init__(self, capacity: int):
        self.capacity: int = capacity
        self.cache: Dict[str, Node] = {}
        self.head: Node = Node("head", None)
        self.tail: Node = Node("tail", None)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node) -> None:
        p, n = node.prev, node.next
        if p and n:
            p.next = n
            n.prev = p

    def _add_to_head(self, node: Node) -> None:
        nxt = self.head.next
        self.head.next = node
        node.prev = self.head
        node.next = nxt
        if nxt:
            nxt.prev = node

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            self._add_to_head(node)
            return node.val
        return None

    def put(self, key: str, val: Any) -> Optional[str]:
        """Inserts item. Returns evicted key if eviction occurs, otherwise None."""
        evicted_key = None
        if key in self.cache:
            self._remove(self.cache[key])
        
        new_node = Node(key, val)
        self.cache[key] = new_node
        self._add_to_head(new_node)
        
        if len(self.cache) > self.capacity:
            lru_node = self.tail.prev
            if lru_node and lru_node != self.head:
                evicted_key = lru_node.key
                self._remove(lru_node)
                del self.cache[evicted_key]
        return evicted_key

class ContextDAG:
    """Custom Directed Acyclic Graph tracking context dependencies between agents."""
    def __init__(self):
        self.adjacency_list: Dict[str, List[str]] = {}
        self.in_degree: Dict[str, int] = {}
        self.node_payloads: Dict[str, Dict[str, Any]] = {}

    def add_node(self, node_id: str, payload: Dict[str, Any]) -> None:
        if node_id not in self.adjacency_list:
            self.adjacency_list[node_id] = []
            self.in_degree[node_id] = 0
            self.node_payloads[node_id] = payload

    def add_edge(self, source: str, destination: str) -> None:
        self.add_node(source, {})
        self.add_node(destination, {})
        if destination not in self.adjacency_list[source]:
            self.adjacency_list[source].append(destination)
            self.in_degree[destination] += 1