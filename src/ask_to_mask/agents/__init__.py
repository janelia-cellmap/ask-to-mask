"""Two-agent refinement loop for iterative mask quality improvement."""

from .gen_backend import (
    FluxBackend,
    GeminiImageBackend,
    GLMImageBackend,
    ImageGenBackend,
    QwenImageEditBackend,
    create_gen_backend,
)
from .sam3_backend import SAM3Backend
from .llm_backend import LLMBackend, create_llm_backend
from .loop import LoopConfig, LoopResult, run_refinement_loop
from .zstack import ZStackResult, run_zstack_refinement
from .schemas import (
    DetailedScores,
    GenerationParams,
    GenerationResult,
    EvaluationResult,
)

__all__ = [
    "ImageGenBackend",
    "FluxBackend",
    "GeminiImageBackend",
    "GLMImageBackend",
    "QwenImageEditBackend",
    "SAM3Backend",
    "create_gen_backend",
    "LLMBackend",
    "create_llm_backend",
    "LoopConfig",
    "LoopResult",
    "run_refinement_loop",
    "ZStackResult",
    "run_zstack_refinement",
    "DetailedScores",
    "GenerationParams",
    "GenerationResult",
    "EvaluationResult",
]
