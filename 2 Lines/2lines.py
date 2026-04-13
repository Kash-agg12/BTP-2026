import os
import imageio
import pandas as pd

# =====================================================
# FILES
# =====================================================
CSV_FILE = "/data/sriparna/siddharth2201cs86/full_dataset_processed.csv"
OUTPUT_DIR = "full_outputs_2_lines"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# LOAD CSV
# =====================================================
df = pd.read_csv(CSV_FILE)
print("Loaded", len(df), "poems\n")

# =====================================================
# CHECK IF GENERATION IS NEEDED
# =====================================================
need_generation = False

for i, row in df.iterrows():

    title = str(row["Title"]).replace(" ", "_")
    poet = str(row["Poet"]).replace(" ", "_")
    folder_name = f"{i}_{title}_{poet}"
    poem_output_dir = os.path.join(OUTPUT_DIR, folder_name)

    if not os.path.exists(poem_output_dir):
        need_generation = True
        break

    existing_images = [
        f for f in os.listdir(poem_output_dir)
        if f.startswith("scene_") and f.endswith(".png")
    ]

    if len(existing_images) == 0:
        need_generation = True
        break

# =====================================================
# VIDEO-ONLY MODE
# =====================================================
FPS = 24  # 1 second per frame

if not need_generation:

    print("All images exist → running video-only mode\n")

    for i, row in df.iterrows():

        title = str(row["Title"]).replace(" ", "_")
        poet = str(row["Poet"]).replace(" ", "_")

        folder_name = f"{i}_{title}_{poet}"
        poem_output_dir = os.path.join(OUTPUT_DIR, folder_name)

        existing_images = sorted([
            f for f in os.listdir(poem_output_dir)
            if f.startswith("scene_") and f.endswith(".png")
        ])

        if len(existing_images) == 0:
            continue

        print("Creating video for:", title)

        frames = []

        for img_name in existing_images:
            path = os.path.join(poem_output_dir, img_name)
            frame = imageio.imread(path)

            # 👉 1 second per image
            for _ in range(FPS):
                frames.append(frame)

        video_path = os.path.join(poem_output_dir, "story.mp4")
        imageio.mimsave(video_path, frames, fps=FPS)

        print("Video saved:", video_path)

    print("\nAll videos created.")
    exit()

# =====================================================
# FULL PIPELINE (ONLY IF NEEDED)
# =====================================================
print("Generation required → loading models...\n")

import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import spacy

nlp = spacy.load("en_core_web_sm")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

generator = torch.Generator(device=DEVICE).manual_seed(42)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=DTYPE,
    use_safetensors=True
)

pipe.to(DEVICE)
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

img2img = StableDiffusionXLImg2ImgPipeline(**pipe.components)

# =====================================================
# STYLE
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
# GENERATION FUNCTION
# =====================================================
def generate_poem_video(poem_text, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    lines = [l.strip() for l in poem_text.split("\n") if l.strip()]

    segments = [
        " ".join(lines[i:i+2])
        for i in range(0, len(lines), 2)
    ]

    if len(segments) == 0:
        return

    segments = segments[:12]

    poem_context = " ".join(segments[:3])

    style_anchor = pipe(
        prompt=f"cinematic visual interpretation of: {poem_context}",
        guidance_scale=7.5,
        num_inference_steps=30,
        height=1024,
        width=1024,
        generator=generator
    ).images[0]

    style_anchor.save(os.path.join(output_dir, "anchor.png"))

    generated_paths = []
    previous_image = style_anchor

    for idx, segment in enumerate(segments):

        prompt = f"{GLOBAL_STYLE}, {BASE_STYLE}, {build_structured_prompt(segment)}"

        result = img2img(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=previous_image,
            strength=0.6 if idx == 0 else 0.5,
            guidance_scale=9.0,
            num_inference_steps=35,
            generator=torch.Generator(device=DEVICE).manual_seed(1000 + idx)
        )

        image = result.images[0]
        path = os.path.join(output_dir, f"scene_{idx}.png")
        image.save(path)

        generated_paths.append(path)
        previous_image = image

    # 👉 VIDEO (1 sec per image)
    frames = []

    for path in generated_paths:
        frame = imageio.imread(path)

        for _ in range(FPS):
            frames.append(frame)

    imageio.mimsave(os.path.join(output_dir, "story.mp4"), frames, fps=FPS)


# =====================================================
# MAIN LOOP
# =====================================================
for i, row in df.iterrows():

    title = str(row["Title"]).replace(" ", "_")
    poet = str(row["Poet"]).replace(" ", "_")

    poem_text = str(row["ctext"])

    folder_name = f"{i}_{title}_{poet}"
    poem_output_dir = os.path.join(OUTPUT_DIR, folder_name)

    existing_images = []
    if os.path.exists(poem_output_dir):
        existing_images = [
            f for f in os.listdir(poem_output_dir)
            if f.startswith("scene_") and f.endswith(".png")
        ]

    if len(existing_images) > 0:
        print("Skipping generation (images exist):", title)
        continue

    print("Generating:", title)
    generate_poem_video(poem_text, poem_output_dir)

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

print("\nAll poems processed.")