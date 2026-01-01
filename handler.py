import runpod
import torch
import os
import base64
import tempfile
import imageio
from PIL import Image
from io import BytesIO

# Initialize model globally (loaded once)
model = None
pipe = None

def load_model():
    global pipe
    if pipe is not None:
        return pipe
    
    from diffusers import DiffusionPipeline
    
    pipe = DiffusionPipeline.from_pretrained(
        "/app/models/wan",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")
    
    # Enable memory optimizations
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    
    return pipe

def handler(job):
    """RunPod serverless handler for video generation"""
    job_input = job["input"]
    
    prompt = job_input.get("prompt", "A beautiful sunset over mountains")
    num_frames = job_input.get("num_frames", 24)  # ~3 seconds at 8fps
    height = job_input.get("height", 512)
    width = job_input.get("width", 512)
    num_inference_steps = job_input.get("num_inference_steps", 25)
    guidance_scale = job_input.get("guidance_scale", 7.5)
    
    # Optional: Image-to-video
    init_image = job_input.get("init_image")  # Base64 encoded
    
    try:
        pipe = load_model()
        
        # Generate video frames
        if init_image:
            # Decode base64 image
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
                guidance_scale=guidance_scale,
            )
        else:
            output = pipe(
                prompt=prompt,
                num_frames=num_frames,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
        
        # Convert frames to video
        frames = output.frames[0]  # List of PIL Images
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            writer = imageio.get_writer(tmp.name, fps=8, codec="libx264")
            for frame in frames:
                writer.append_data(frame)
            writer.close()
            
            # Read and encode video
            with open(tmp.name, "rb") as f:
                video_base64 = base64.b64encode(f.read()).decode("utf-8")
            
            os.unlink(tmp.name)
        
        return {
            "status": "success",
            "video": video_base64,
            "format": "mp4",
            "frames": num_frames,
            "resolution": f"{width}x{height}"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

runpod.serverless.start({"handler": handler})
