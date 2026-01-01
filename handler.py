import runpod
import torch
import os
import base64
import tempfile
import imageio
from PIL import Image
from io import BytesIO
import subprocess

pipe = None

def setup():
    subprocess.run(
        "pip install diffusers transformers accelerate safetensors "
        "imageio imageio-ffmpeg opencv-python-headless pillow huggingface_hub",
        shell=True,
        check=True
    )

    if not os.path.exists("/app/models/wan"):
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="alibaba-pai/Wan2.2-T2V-1.3B",
            local_dir="/app/models/wan",
            local_dir_use_symlinks=False
        )

def load_model():
    global pipe
    if pipe is not None:
        return pipe

    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(
        "/app/models/wan",
        torch_dtype=torch.float16
    ).to("cuda")

    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()

    return pipe

def handler(job):
    job_input = job["input"]

    prompt = job_input.get("prompt", "A cinematic sunset over mountains")
    num_frames = job_input.get("num_frames", 24)
    height = job_input.get("height", 512)
    width = job_input.get("width", 512)
    num_inference_steps = job_input.get("num_inference_steps", 25)
    guidance_scale = job_input.get("guidance_scale", 7.5)
    init_image = job_input.get("init_image")

    try:
        load_model()

        if init_image:
            image_data = base64.b64decode(init_image)
            image = Image.open(BytesIO(image_data)).convert("RGB")
            image = image.resize((width, height))

            output = pipe(
                prompt=prompt,
                image=image,
                num_frames=num_frames,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
        else:
            output = pipe(
                prompt=prompt,
                num_frames=num_frames,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )

        frames = output.frames[0]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            writer = imageio.get_writer(tmp.name, fps=8, codec="libx264")
            for frame in frames:
                writer.append_data(frame)
            writer.close()

            with open(tmp.name, "rb") as f:
                video_base64 = base64.b64encode(f.read()).decode()

            os.unlink(tmp.name)

        return {
            "status": "success",
            "video": video_base64,
            "format": "mp4",
            "frames": num_frames,
            "resolution": f"{width}x{height}"
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}

setup()
runpod.serverless.start({"handler": handler})
