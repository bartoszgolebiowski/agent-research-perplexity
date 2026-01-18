from __future__ import annotations

"""Queue management for analysis nodes."""

from collections import deque
from typing import Deque, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import AnalysisNode, ICPAnalysisInput

from src.engine.types import AnalysisNodeStatus


class AnalysisQueue:
    """
    Manages the execution queue for analysis nodes.
    
    Ensures parent nodes are processed before children and supports
    dynamic node insertion for discovered sub-tasks.
    """

    def __init__(self) -> None:
        self._queue: Deque[str] = deque()
        self._nodes: dict[str, AnalysisNode] = {}
        self._input: Optional[ICPAnalysisInput] = None

    @classmethod
    def from_input(cls, input_spec: ICPAnalysisInput) -> "AnalysisQueue":
        """Create a queue from an input specification."""
        queue = cls()
        queue._input = input_spec
        
        # Build node lookup and queue in tree order
        all_nodes = input_spec.get_all_nodes()
        for node in all_nodes:
            queue._nodes[node.id] = node
            if node.status == AnalysisNodeStatus.PENDING:
                queue._queue.append(node.id)
        
        return queue

    def get_next_pending(self) -> Optional[AnalysisNode]:
        """
        Get the next pending node from the queue.
        
        Skips nodes that are no longer pending (may have been updated).
        """
        while self._queue:
            node_id = self._queue[0]
            node = self._nodes.get(node_id)
            if node and node.status == AnalysisNodeStatus.PENDING:
                return node
            # Remove non-pending nodes from queue
            self._queue.popleft()
        return None

    def mark_in_progress(self, node_id: str) -> None:
        """Mark a node as in-progress and remove from queue."""
        if node_id in self._nodes:
            self._nodes[node_id].status = AnalysisNodeStatus.IN_PROGRESS
        if self._queue and self._queue[0] == node_id:
            self._queue.popleft()

    def mark_completed(self, node_id: str) -> None:
        """Mark a node as completed."""
        if node_id in self._nodes:
            self._nodes[node_id].status = AnalysisNodeStatus.COMPLETED

    def mark_partial(self, node_id: str) -> None:
        """Mark a node as partially completed."""
        if node_id in self._nodes:
            self._nodes[node_id].status = AnalysisNodeStatus.PARTIAL

    def mark_failed(self, node_id: str) -> None:
        """Mark a node as failed."""
        if node_id in self._nodes:
            self._nodes[node_id].status = AnalysisNodeStatus.FAILED

    def mark_skipped(self, node_id: str) -> None:
        """Mark a node as skipped."""
        if node_id in self._nodes:
            self._nodes[node_id].status = AnalysisNodeStatus.SKIPPED

    def requeue_for_retry(self, node_id: str) -> None:
        """Re-add a node to the front of the queue for retry."""
        if node_id in self._nodes:
            node = self._nodes[node_id]
            node.status = AnalysisNodeStatus.PENDING
            node.attempt_count += 1
            # Add to front of queue for immediate retry
            self._queue.appendleft(node_id)

    def add_dynamic_node(self, node: AnalysisNode, parent_id: str) -> None:
        """
        Add a dynamically discovered node to the queue.
        
        The node is added after its parent's position in processing order.
        """
        node.parent_id = parent_id
        self._nodes[node.id] = node
        
        # Also add to the parent's sub_tasks if input is available
        if self._input:
            parent = self._input.get_node_by_id(parent_id)
            if parent:
                parent.sub_tasks.append(node)
        
        # Add to queue
        self._queue.append(node.id)

    def get_node(self, node_id: str) -> Optional[AnalysisNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def update_node(self, node: AnalysisNode) -> None:
        """Update a node in the queue."""
        self._nodes[node.id] = node

    def get_all_nodes(self) -> List[AnalysisNode]:
        """Get all nodes in the queue."""
        return list(self._nodes.values())

    def is_empty(self) -> bool:
        """Check if there are no more pending nodes."""
        return self.get_next_pending() is None

    def pending_count(self) -> int:
        """Count remaining pending nodes."""
        return sum(
            1 for n in self._nodes.values() 
            if n.status == AnalysisNodeStatus.PENDING
        )

    def get_input(self) -> Optional[ICPAnalysisInput]:
        """Get the original input specification."""
        return self._input
