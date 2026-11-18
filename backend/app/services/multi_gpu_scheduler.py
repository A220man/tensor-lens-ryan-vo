"""Heterogeneous Multi-GPU Pipeline Scheduler."""
from typing import Dict, List, Any
from app.models.entities import ModelSpecRecord, HardwareProfileRecord
from app.services.memory_planner import MemoryPlanner


class MultiGPUScheduler:
    """Distribute transformer layers across a set of heterogeneous GPUs."""

    @classmethod
    def distribute_layers(
        cls,
        model: ModelSpecRecord,
        hardware_list: List[HardwareProfileRecord],
        quantization: str = "int4",
        batch_size: int = 1,
        seq_len: int = 512,
    ) -> Dict[str, List[int]]:
        """
        Greedy assignment of model layers to GPUs based on their capacity.
        Returns a dict mapping hardware.id -> list of assigned layer indices.
        """
        # Compute maximum layers each GPU can hold
        hw_capacities: Dict[str, int] = {}
        for hw in hardware_list:
            # Use manual_chunk_size = model.num_layers to let solve_optimal_chunk_size
            # return the largest chunk size that fits into hw's VRAM.
            plan = MemoryPlanner.solve_optimal_chunk_size(
                model=model,
                hardware=hw,
                quantization=quantization,
                batch_size=batch_size,
                seq_len=seq_len,
                manual_chunk_size=model.num_layers,
            )
            max_layers = plan["layers_per_chunk"]
            hw_capacities[hw.id] = max_layers

        # Sort GPUs by capacity (larger first) for better utilization
        sorted_hw = sorted(hardware_list, key=lambda h: hw_capacities[h.id], reverse=True)

        assignments: Dict[str, List[int]] = {}
        remaining_layers = model.num_layers
        next_layer = 0
        for hw in sorted_hw:
            max_layers = hw_capacities[hw.id]
            assign = min(remaining_layers, max_layers)
            if assign <= 0:
                continue
            layers = list(range(next_layer, next_layer + assign))
            assignments[hw.id] = layers
            remaining_layers -= assign
            next_layer += assign

        # Record unassigned layers (should be zero if total capacity >= model.num_layers)
        # If not, we could optionally log or raise a warning.
        return assignments
