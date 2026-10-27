from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    app_name: str = "Model Runner"
    models_config_path: Path = ROOT / "config/models.yaml"
    outputs_dir: Path = ROOT / "outputs"
    default_image_model: str = "sdxl-placeholder"
    default_video_model: str = "wan2.2-ti2v-5b"
    # Needed on 24GB GPUs; turn off on ~40GB+ (A40, A100, H100) for speed.
    cpu_offload: bool = True

    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")


settings = Settings()
