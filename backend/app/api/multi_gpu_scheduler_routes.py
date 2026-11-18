"""Multi-GPU Scheduler API routes."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.auth import get_current_user
from app.models.entities import ModelSpecRecord, HardwareProfileRecord
from app.models.schemas import MultiGPUScheduleRequest, MultiGPUScheduleResponse
from app.services.multi_gpu_scheduler import MultiGPUScheduler

multi_gpu_scheduler_router = APIRouter(prefix="/api/scheduler")

@multi_gpu_scheduler_router.post("/assign", response_model=MultiGPUScheduleResponse)
def assign_layers_to_gpus(
    req: MultiGPUScheduleRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
) -> MultiGPUScheduleResponse:
    """Distribute model layers across heterogeneous GPUs based on capacity."""
    model = db.query(ModelSpecRecord).filter(ModelSpecRecord.id == req.model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    hw_records = db.query(HardwareProfileRecord).filter(
        HardwareProfileRecord.id.in_(req.hardware_ids)
    ).all()
    if len(hw_records) != len(req.hardware_ids):
        raise HTTPException(status_code=404, detail="One or more hardware profiles not found")

    scheduler = MultiGPUScheduler()
    assignments = scheduler.distribute_layers(
        model=model,
        hardware_list=hw_records,
        quantization=req.quantization,
        batch_size=req.batch_size,
        seq_len=req.seq_len,
    )
    total_assigned = sum(len(layers) for layers in assignments.values())
    unassigned = model.num_layers - total_assigned
    return MultiGPUScheduleResponse(
        assignments=assignments,
        total_layers_assigned=total_assigned,
        unassigned_layers=unassigned,
    )
