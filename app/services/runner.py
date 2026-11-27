import io
import logging
import math
import queue
import random
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from fastapi import HTTPException

from app.config import settings
from app.schemas import ModelInfo

log = logging.getLogger(__name__)


class DiffusersImageAdapter:
    def __init__(self, info: ModelInfo) -> None:
        self.info = info

    def generate(self, **kwargs: Any) -> dict[str, Any]:
        # Load the pipeline lazily in __init__ or on first call and keep it; don't reload per request.
        raise NotImplementedError("Image generation is not configured yet. Implement DiffusersImageAdapter.")


class WanAdapter:
    """Wan2.2 video. One checkpoint, one or both modes (TI2V-5B does both from shared weights)."""

    # ponytail: one video model resident at a time; switching models reloads weights (minutes).
    # Keep several loaded only if the GPU and RAM can hold them all.
    _active: "WanAdapter | None" = None

    def __init__(self, info: ModelInfo) -> None:
        self.info = info
        self._t2v = None
        self._i2v = None
        self.offload = settings.cpu_offload if info.cpu_offload is None else info.cpu_offload

    @property
    def loaded(self) -> bool:
        return self._t2v is not None or self._i2v is not None

    def _pipelines(self) -> tuple[Any, Any, dict[str, Any]]:
        """(text-to-video class, image-to-video class, from_pretrained kwargs)."""
        import torch
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline

        vae = AutoencoderKLWan.from_pretrained(self.info.model_id, subfolder="vae", dtype=torch.float32)
        return WanPipeline, WanImageToVideoPipeline, {"vae": vae}

    def _load(self) -> None:
        import torch

        t2v_cls, i2v_cls, kwargs = self._pipelines()
        if "t2v" in self.info.modes:
            self._t2v = t2v_cls.from_pretrained(self.info.model_id, dtype=torch.bfloat16, **kwargs)
            if "i2v" in self.info.modes:
                # Shares every module with _t2v, so no extra memory.
                self._i2v = i2v_cls.from_pipe(self._t2v)
        else:
            self._i2v = i2v_cls.from_pretrained(self.info.model_id, dtype=torch.bfloat16, **kwargs)
        if not self.offload:
            (self._t2v or self._i2v).to("cuda")

    def unload(self) -> None:
        import gc

        import torch

        self._t2v = self._i2v = None
        gc.collect()
        torch.cuda.empty_cache()

    def _run(self, pipe: Any, kwargs: dict[str, Any], guidance_scale: float, tick: Callable[[int], None]) -> Any:
        def on_step(_pipe: Any, i: int, _t: Any, callback_kwargs: dict) -> dict:
            tick(i + 1)
            return callback_kwargs

        return pipe(**kwargs, guidance_scale=guidance_scale, callback_on_step_end=on_step).frames[0]

    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        image: bytes | None,
        num_frames: int,
        fps: int,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        seed: int | None,
        report: Callable[..., None],
    ) -> dict[str, Any]:
        import torch
        from diffusers.utils import export_to_video
        from PIL import Image

        kwargs: dict[str, Any] = {}
        if image is not None:
            try:
                img = Image.open(io.BytesIO(image)).convert("RGB")
            except OSError as exc:
                raise HTTPException(422, "image is not a readable image file") from exc
            # Keep the requested pixel area, follow the image's aspect ratio; sides must be multiples of 32.
            area, ratio = width * height, img.height / img.width
            height = max(32, round(math.sqrt(area * ratio)) // 32 * 32)
            width = max(32, round(math.sqrt(area / ratio)) // 32 * 32)
            kwargs["image"] = img.resize((width, height))

        seed = random.randrange(2**32) if seed is None else seed
        if not self.loaded:
            if WanAdapter._active not in (None, self):
                report(phase="unloading previous model")
                WanAdapter._active.unload()
            report(phase="loading model")
            self._load()
            WanAdapter._active = self
        pipe = self._i2v if image is not None else self._t2v
        if self.offload:
            # TI2V's two pipelines share modules; re-hook them for the one about to run.
            pipe.enable_model_cpu_offload()

        start = time.time()

        def tick(done: int) -> None:
            eta = (time.time() - start) / done * (steps - done)
            report(phase="denoising", step=done, total_steps=steps, eta_seconds=round(eta, 1))

        report(phase="denoising", step=0, total_steps=steps, eta_seconds=None)
        torch.cuda.reset_peak_memory_stats()
        kwargs.update(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=steps,
            generator=torch.Generator("cpu").manual_seed(seed),
        )
        frames = self._run(pipe, kwargs, guidance_scale, tick)
        denoise_seconds = time.time() - start
        peak_vram_mb = torch.cuda.max_memory_allocated() // 2**20

        report(phase="encoding video")
        path = settings.outputs_dir / f"{uuid.uuid4().hex}.mp4"
        export_to_video(frames, str(path), fps=fps)
        # Read the size back: some pipelines pick their own (HunyuanVideo 1.5 image-to-video).
        height, width = frames[0].shape[:2]
        return {
            "video_url": f"/outputs/{path.name}",
            "file_size_bytes": path.stat().st_size,
            "denoise_seconds": round(denoise_seconds, 1),
            "seconds_per_step": round(denoise_seconds / steps, 2),
            "peak_vram_mb": peak_vram_mb,
            "mode": "image-to-video" if image is not None else "text-to-video",
            "width": width,
            "height": height,
            "num_frames": len(frames),
            "fps": fps,
            "seed": seed,
        }


class HunyuanVideo15Adapter(WanAdapter):
    """HunyuanVideo 1.5: separate checkpoints for text-to-video and image-to-video."""

    def _pipelines(self) -> tuple[Any, Any, dict[str, Any]]:
        from diffusers import HunyuanVideo15ImageToVideoPipeline, HunyuanVideo15Pipeline

        return HunyuanVideo15Pipeline, HunyuanVideo15ImageToVideoPipeline, {}

    def _load(self) -> None:
        super()._load()
        (self._t2v or self._i2v).vae.enable_tiling()

    def _run(self, pipe: Any, kwargs: dict[str, Any], guidance_scale: float, tick: Callable[[int], None]) -> Any:
        if "image" in kwargs:
            # The image-to-video pipeline sizes the video from the image itself.
            del kwargs["height"], kwargs["width"]
        # No guidance_scale or step callback in this pipeline: set the guider, count scheduler steps.
        pipe.guider.guidance_scale = guidance_scale
        sched, done = pipe.scheduler, 0

        def step(*args: Any, **kw: Any) -> Any:
            nonlocal done
            out = type(sched).step(sched, *args, **kw)
            done += 1
            tick(done)
            return out

        sched.step = step
        try:
            return pipe(**kwargs).frames[0]
        finally:
            del sched.step


ADAPTERS = {
    "diffusers_image": DiffusersImageAdapter,
    "wan": WanAdapter,
    "hunyuan_video15": HunyuanVideo15Adapter,
}


class ModelRegistry:
    def __init__(self) -> None:
        self.models: dict[str, ModelInfo] = {}
        self._instances: dict[str, Any] = {}

    def load(self, path: Path) -> None:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        models = [ModelInfo(**item) for item in data.get("models", [])]
        for m in models:
            if m.adapter not in ADAPTERS:
                raise RuntimeError(f"Model {m.id}: unknown adapter '{m.adapter}'")
        self.models = {m.id: m for m in models}
        self._instances.clear()

    def resolve(self, model_id: str | None, task: str):
        if model_id is None:
            model_id = settings.default_image_model if task == "image" else settings.default_video_model
        info = self.models.get(model_id)
        if info is None:
            raise HTTPException(404, f"Unknown model: {model_id}")
        if info.task != task:
            raise HTTPException(400, f"Model {model_id} does not support task '{task}'")
        if not info.enabled:
            raise HTTPException(503, f"Model {model_id} is disabled in config")
        if model_id not in self._instances:
            self._instances[model_id] = ADAPTERS[info.adapter](info)
        return self._instances[model_id]

    def is_loaded(self, model_id: str) -> bool:
        return getattr(self._instances.get(model_id), "loaded", False)


registry = ModelRegistry()


# ponytail: in-memory, so queued/running jobs are lost on restart (finished videos stay on disk).
# Move job records to SQLite if restarts become routine.
class JobQueue:
    """FIFO with a single worker thread: exactly one job on the GPU at a time."""

    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._work, daemon=True).start()

    def submit(self, adapter: Any, kwargs: dict[str, Any]) -> str:
        job_id = uuid.uuid4().hex
        self.jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "model": adapter.info.id,
            # Request params as sent, minus raw image bytes.
            "params": {k: ({"bytes": len(v)} if isinstance(v, bytes) else v) for k, v in kwargs.items()},
            "progress": None,
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
            "_adapter": adapter,
            "_kwargs": kwargs,
        }
        self._queue.put(job_id)
        return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        job = self.jobs.get(job_id)
        if job is None:
            return None
        # dict() copies atomically; iterating the live dict races with the worker popping keys.
        view = {k: v for k, v in dict(job).items() if not k.startswith("_")}
        now = time.time()
        view["wait_seconds"] = round((view["started_at"] or now) - view["created_at"], 1)
        if view["started_at"]:
            view["run_seconds"] = round((view["finished_at"] or now) - view["started_at"], 1)
        if view["status"] == "queued":
            # Jobs waiting ahead of this one; 0 = next up.
            view["position"] = [j["id"] for j in list(self.jobs.values()) if j["status"] == "queued"].index(job_id)
        return view

    def list(self, status: str | None = None) -> list[dict[str, Any]]:
        views = [self.get(job_id) for job_id in reversed(list(self.jobs))]
        return [v for v in views if status is None or v["status"] == status]

    def _work(self) -> None:
        while True:
            job = self.jobs[self._queue.get()]
            job.update(status="running", started_at=time.time())

            def report(_job: dict = job, **info: Any) -> None:
                _job["progress"] = info

            try:
                # Pop so request data (e.g. uploaded images) isn't kept in memory forever.
                job["result"] = job.pop("_adapter").generate(**job.pop("_kwargs"), report=report)
                job["status"] = "completed"
            except Exception as exc:
                log.exception("Job %s failed", job["id"])
                job["error"] = getattr(exc, "detail", None) or str(exc)
                job["status"] = "failed"
            job["finished_at"] = time.time()


jobs = JobQueue()
