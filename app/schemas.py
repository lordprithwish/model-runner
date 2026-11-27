from typing import Literal

from pydantic import Base64Bytes, BaseModel, Field, model_validator

from app.config import settings


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
    # Rounded to the nearest 4k+1 (5, 9, ..., 121): the models compress time 4x.
    num_frames: int = Field(default=121, ge=5, description="At most MAX_FRAMES (default 241)")
    fps: int = Field(default=24, ge=1, le=60)
    # Alternative to num_frames: video length at `fps`. Wins when both are sent.
    seconds: float | None = Field(default=None, gt=0)
    # With an image these set the pixel area; the image's aspect ratio decides the shape.
    width: int = Field(default=1280, ge=256, le=1280, multiple_of=32)
    height: int = Field(default=704, ge=256, le=1280, multiple_of=32)
    steps: int = Field(default=50, ge=1, le=100)
    guidance_scale: float = Field(default=5.0, ge=1.0, le=20.0)
    seed: int | None = Field(default=None, ge=0, lt=2**32)

    @model_validator(mode="after")
    def _frames(self) -> "GenerateVideoRequest":
        n = round(self.seconds * self.fps) + 1 if self.seconds is not None else self.num_frames
        self.num_frames = max(5, round((n - 1) / 4) * 4 + 1)
        if self.num_frames > settings.max_frames:
            raise ValueError(f"num_frames {self.num_frames} is over MAX_FRAMES ({settings.max_frames})")
        return self


class ModelInfo(BaseModel):
    id: str
    name: str = ""
    task: Literal["image", "video"]
    adapter: str
    enabled: bool = False
    model_id: str = ""
    # Video only: t2v = text-to-video, i2v = image-to-video.
    modes: list[Literal["t2v", "i2v"]] = ["t2v", "i2v"]
    # None = use the global CPU_OFFLOAD setting.
    cpu_offload: bool | None = None
    notes: str = ""
