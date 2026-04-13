import os
import torch
import imageio
import numpy as np
import pandas as pd
from PIL import Image
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import spacy

# =====================================================
# LOAD NLP
# =====================================================
nlp = spacy.load("en_core_web_sm")

# =====================================================
# MEMORY SETTINGS
# =====================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# =====================================================
# FILES
# =====================================================
CSV_FILE = "/data/sriparna/poem_videos/poems/poem1/images/poem_project/poemsum_valid.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# FIXED SEED
# =====================================================
generator = torch.Generator(device=DEVICE).manual_seed(42)

# =====================================================
# LOAD SDXL
# =====================================================
print("Loading SDXL model...")

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=DTYPE,
    use_safetensors=True
)

pipe.to(DEVICE)

pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

img2img = StableDiffusionXLImg2ImgPipeline(**pipe.components)

print("Model loaded successfully.\n")

# =====================================================
# LOAD CSV
# =====================================================
df = pd.read_csv(CSV_FILE)

print("Loaded", len(df), "poems\n")

# =====================================================
# GLOBAL STYLE
# =====================================================
GLOBAL_STYLE = (
    "cinematic poetic visual storytelling, "
    "consistent character design, "
    "consistent lighting, natural color grading"
)

BASE_STYLE = (
    "ultra detailed, volumetric lighting, "
    "35mm lens photography, realistic, high quality"
)

NEGATIVE_PROMPT = (
    "blurry, low quality, oversaturated, jpeg artifacts, "
    "bad anatomy, malformed limbs, extra fingers, extra arms, "
    "mutated body, fused characters, duplicated character, "
    "cloned animal, same species repeated"
)

# =====================================================
# ENTITY EXTRACTION
# =====================================================
STOP_NOUNS = {
    "day", "time", "thing", "moment", "way",
    "light", "dark", "scene", "background"
}

def extract_entities(text, max_entities=5):

    doc = nlp(text)

    entities = []

    for chunk in doc.noun_chunks:

        phrase = chunk.text.lower().strip()

        if phrase.startswith("the "):
            phrase = phrase[4:]

        if phrase not in STOP_NOUNS:
            entities.append(phrase)

    seen = set()
    unique = []

    for e in entities:
        if e not in seen and len(e) > 2:
            unique.append(e)
            seen.add(e)

    return unique[:max_entities]


def build_structured_prompt(segment):

    entities = extract_entities(segment)

    if len(entities) >= 2:

        entity_phrase = " and ".join(
            [f"(one {e}:1.5)" for e in entities[:2]]
        )

        return f"{entity_phrase}, two different species. {segment}"

    return segment


# =====================================================
# VIDEO GENERATION FUNCTION
# =====================================================
def generate_poem_video(poem_text, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    segments = [l.strip() for l in poem_text.split("\n") if l.strip()]

    if len(segments) == 0:
        return

    # limit scenes for speed
    segments = segments[:12]

    print("Segments:", len(segments))

    # =====================================================
    # STEP 1 — STYLE ANCHOR
    # =====================================================
    poem_context = " ".join(segments[:3])

    style_anchor_prompt = (
        f"cinematic visual interpretation of: {poem_context}, "
        "consistent lighting, natural color grading, realistic photography"
    )

    style_anchor = pipe(
        prompt=style_anchor_prompt,
        guidance_scale=7.5,
        num_inference_steps=30,
        height=1024,
        width=1024,
        generator=generator
    ).images[0]

    anchor_path = os.path.join(output_dir, "anchor.png")
    style_anchor.save(anchor_path)

    generated_paths = []
    previous_image = style_anchor

    # =====================================================
    # STEP 2 — TEMPORAL SCENES
    # =====================================================
    for idx, segment in enumerate(segments):

        print("Scene", idx + 1, "/", len(segments))
        print(segment)

        structured_segment = build_structured_prompt(segment)

        prompt = (
            f"{GLOBAL_STYLE}, {BASE_STYLE}, "
            f"{structured_segment}, "
            "Maintain species accuracy and visual distinction."
        )

        strength = 0.6 if idx == 0 else 0.5

        scene_generator = torch.Generator(device=DEVICE).manual_seed(1000 + idx)

        result = img2img(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=previous_image,
            strength=strength,
            guidance_scale=9.0,
            num_inference_steps=35,
            generator=scene_generator
        )

        image = result.images[0]

        scene_path = os.path.join(output_dir, f"scene_{idx}.png")
        image.save(scene_path)

        generated_paths.append(scene_path)

        previous_image = image

    # =====================================================
    # STEP 3 — VIDEO
    # =====================================================
    frames = []

    for path in generated_paths:

        frame = imageio.imread(path)

        for _ in range(4):
            frames.append(frame)

    video_path = os.path.join(output_dir, "story.mp4")

    imageio.mimsave(video_path, frames, fps=8)

    print("Video saved:", video_path)


# =====================================================
# MAIN LOOP
# =====================================================
for i, row in df.iterrows():

    title = str(row["Title"]).replace(" ", "_")
    poet = str(row["Poet"]).replace(" ", "_")

    poem_text = str(row["ctext"])

    folder_name = f"{i}_{title}_{poet}"

    poem_output_dir = os.path.join(OUTPUT_DIR, folder_name)

    print("\n====================================")
    print("Generating video for:", title)
    print("====================================\n")

    try:

        generate_poem_video(poem_text, poem_output_dir)

        # prevent GPU memory buildup
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    except Exception as e:

        print("Failed for:", title)
        print(e)

print("\nAll poems processed.")