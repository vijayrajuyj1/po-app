from fastapi import APIRouter, Depends

from apps.ai.schemas import AIGenerationRequestSchema, AIGenerationResponseSchema
from apps.ai.service import AIService
from security.auth_backend import require_roles
from constants.roles import ADMIN


router = APIRouter(prefix="/api/ai", tags=["AI"], dependencies=[Depends(require_roles(ADMIN))])


@router.post("/generate-extraction-prompt", response_model=AIGenerationResponseSchema)
async def generate_extraction_prompt(payload: AIGenerationRequestSchema) -> AIGenerationResponseSchema:
    return await AIService.generate_extraction_prompt(payload)


