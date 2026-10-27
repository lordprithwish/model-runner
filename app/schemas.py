from typing import Literal

from pydantic import Base64Bytes, BaseModel, Field


class GenerateImageRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=10_000)
    negative_prompt: str | None = Field(default=None, max_length=10_000)
    model: str | None = None
    width: int = Field(default=1024, ge=64, le=4096)
    height: int = Field(default=1024, ge=64, le=4096)
    steps: int = Field(default=30, ge=1, le=200)
    seed: int | None = Field(default=None, ge=0)


class GenerateVideoRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=10_000)
    negative_prompt: str | None = Field(default=None, max_length=10_000)
    model: str | None = None
    # First frame for image-to-video, plain base64 (no "data:" prefix). Omit for text-to-video.
    image: Base64Bytes | None = Field(default=None, max_length=20_000_000)
    num_frames: int = Field(default=121, ge=5, le=121)
    fps: int = Field(default=24, ge=1, le=60)
    # With an image these set the pixel area; the image's aspect ratio decides the shape.
    width: int = Field(default=1280, ge=256, le=1280, multiple_of=32)
    height: int = Field(default=704, ge=256, le=1280, multiple_of=32)
    steps: int = Field(default=50, ge=1, le=100)
    guidance_scale: float = Field(default=5.0, ge=1.0, le=20.0)
    seed: int | None = Field(default=None, ge=0, lt=2**32)


class ModelInfo(BaseModel):
    id: str
    name: str = ""
    task: Literal["image", "video"]
    adapter: str
    enabled: bool = False
    model_id: str = ""
    notes: str = ""
