import os
import base64
import json
import io
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY environment variable is not set")

client = genai.Client(api_key=GOOGLE_API_KEY)

app = FastAPI(title="Product Image Replacer")
templates = Jinja2Templates(directory="templates")

GEMINI_IMAGE_GEN_MODEL = "gemini-2.0-flash-preview-image-generation"
GEMINI_ANALYSIS_MODEL = "gemini-2.0-flash"


def normalize_image(image_bytes: bytes, max_size: int = 1536) -> tuple[bytes, str]:
    """Convert image to JPEG and resize if needed, returns (bytes, mime_type)."""
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue(), "image/jpeg"


def replace_product_in_scene(
    product_bytes: bytes,
    scene_bytes: bytes,
    extra_prompt: str = "",
) -> bytes:
    """
    Use Gemini image generation to seamlessly place the reference product into the scene.
    Image 1 (product) + Image 2 (scene) → new scene with product integrated.
    """
    product_norm, prod_mime = normalize_image(product_bytes)
    scene_norm, scene_mime = normalize_image(scene_bytes)

    prompt_parts = [
        "You are given two images: Image 1 is a product reference, Image 2 is a scene.",
        "Generate a new photorealistic version of the scene (Image 2) with the product from Image 1 seamlessly integrated into it.",
        "Preserve the exact visual appearance, branding, colors, shape, and texture of the product from Image 1.",
        "Match the lighting, shadows, and perspective of the scene in Image 2.",
        "The product placement should look completely natural, as if it was always part of the scene.",
        "Output only the final composited scene image.",
    ]
    if extra_prompt.strip():
        prompt_parts.append(extra_prompt.strip())
    prompt = " ".join(prompt_parts)

    response = client.models.generate_content(
        model=GEMINI_IMAGE_GEN_MODEL,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=product_norm, mime_type=prod_mime),
                    types.Part.from_bytes(data=scene_norm, mime_type=scene_mime),
                    types.Part.from_text(text=prompt),
                ],
            )
        ],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )

    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            return part.inline_data.data

    raise ValueError("No image was returned by the model.")


ANALYSIS_PROMPT = """You are an expert image quality analyst specializing in product photography and AI-generated composites.

You are given TWO images:
- Image 1: The REFERENCE product image (the source product to be replicated)
- Image 2: The GENERATED image (the scene with the product integrated by AI)

Analyze how accurately and seamlessly the product from Image 1 has been integrated into Image 2.

Evaluate on these five dimensions (score each 0–100):

1. **product_fidelity** – How faithfully the product's visual identity (shape, color, branding, texture, details) is preserved in the generated image.
2. **lighting_consistency** – How well the product's lighting, highlights, and shadows match the scene's light sources.
3. **perspective_accuracy** – How correctly the product's viewing angle, scale, and foreshortening fit the scene geometry.
4. **edge_integration** – How clean and natural the product's edges and boundaries blend with the scene (no harsh cutouts or halos).
5. **overall_seamlessness** – An overall holistic score for how naturally the product feels part of the scene.

Return ONLY a valid JSON object in exactly this format (no markdown, no extra text):
{
  "scores": {
    "product_fidelity": <0-100>,
    "lighting_consistency": <0-100>,
    "perspective_accuracy": <0-100>,
    "edge_integration": <0-100>,
    "overall_seamlessness": <0-100>
  },
  "overall": <average of the five scores, rounded to nearest integer>,
  "strengths": ["<specific strength 1>", "<specific strength 2>", "<specific strength 3>"],
  "improvements": ["<specific issue 1>", "<specific issue 2>"],
  "summary": "<2-3 sentence overall assessment>"
}"""


def analyze_accuracy(product_bytes: bytes, generated_bytes: bytes) -> dict:
    """Use Gemini vision to compare the reference product against the generated image."""
    product_norm, prod_mime = normalize_image(product_bytes)
    generated_norm, gen_mime = normalize_image(generated_bytes)

    response = client.models.generate_content(
        model=GEMINI_ANALYSIS_MODEL,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=product_norm, mime_type=prod_mime),
                    types.Part.from_bytes(data=generated_norm, mime_type=gen_mime),
                    types.Part.from_text(text=ANALYSIS_PROMPT),
                ],
            )
        ],
    )

    raw = response.text.strip()
    # Strip markdown fences if Gemini wraps the JSON
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/replace")
async def api_replace(
    product_image: UploadFile = File(...),
    scene_image: UploadFile = File(...),
    extra_prompt: str = "",
):
    """Generate a scene image with the reference product seamlessly integrated."""
    try:
        product_bytes = await product_image.read()
        scene_bytes = await scene_image.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded files: {exc}")

    try:
        generated_bytes = replace_product_in_scene(product_bytes, scene_bytes, extra_prompt)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {exc}")

    generated_b64 = base64.b64encode(generated_bytes).decode()
    return JSONResponse({"generated_image": generated_b64})


@app.post("/api/analyze")
async def api_analyze(
    product_image: UploadFile = File(...),
    generated_image: UploadFile = File(...),
):
    """Analyze how accurately the product was integrated into the generated image."""
    try:
        product_bytes = await product_image.read()
        generated_bytes = await generated_image.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded files: {exc}")

    try:
        analysis = analyze_accuracy(product_bytes, generated_bytes)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Analysis response was not valid JSON: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    return JSONResponse(analysis)
