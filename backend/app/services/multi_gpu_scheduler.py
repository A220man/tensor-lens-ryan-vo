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
        Distribute model layers across heterogeneous GPUs based on VRAM capacity.
        Uses proportional allocation: each GPU receives a share of layers
        proportional to its available VRAM, ensuring all GPUs receive work
        and total assigned layers equals model.num_layers.
        Returns a dict mapping hardware.id -> list of assigned layer indices.
        """
        # Compute maximum layers each GPU can hold
        hw_capacities: Dict[str, int] = {}
        for hw in hardware_list:
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

        # Total VRAM capacity across all GPUs for proportional allocation
        total_capacity = sum(hw_capacities.values()) or 1

        # Proportional distribution: each GPU gets a share of layers
        # proportional to its capacity, ensuring total = model.num_layers
        assignments: Dict[str, List[int]] = {}
        remaining_layers = model.num_layers
        next_layer = 0

        for i, hw in enumerate(hardware_list):
            # Proportional share for this GPU
            proportional = int(hw_capacities[hw.id] / total_capacity * remaining_layers)

            # Last GPU gets all remaining layers
            if i == len(hardware_list) - 1:
                assign = remaining_layers
            else:
                # Ensure at least 1 layer per GPU if we have layers remaining
                # and this isn't the last GPU
                assign = max(1, proportional) if proportional > 0 else 1
                assign = min(assign, hw_capacities[hw.id], remaining_layers)

            if assign > 0 and assign <= remaining_layers:
                layers = list(range(next_layer, next_layer + assign))
                assignments[hw.id] = layers
                remaining_layers -= assign
                next_layer += assign

        # Safety: if any layers remain (e.g. from floor constraints),
        # assign them to the last GPU
        if remaining_layers > 0 and hardware_list:
            last_hw = hardware_list[-1]
            if last_hw.id not in assignments:
                assignments[last_hw.id] = list(range(next_layer, next_layer + remaining_layers))
            else:
                assignments[last_hw.id] = list(
                    range(next_layer, next_layer + remaining_layers)
                )

        return assignments
