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
# SEED
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
    "consistent lighting, natural color grading"
)

BASE_STYLE = (
    "ultra detailed, volumetric lighting, "
    "35mm lens photography, realistic, high quality"
)

IDENTITY_LOCK = (
    "same person, same face, same body, same identity, "
    "no transformation, no species change"
)

NEGATIVE_PROMPT = (
    "multiple subjects, duplicate subjects, different species, animal instead of human, "
    "human turning into animal, morphing, transformation, mutation, "
    "extra heads, extra bodies, inconsistent character, identity change, "
    "cloned character, different person, different face, "
    "bad anatomy, blurry, low quality"
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

# =====================================================
# GLOBAL SUBJECT LOCK
# =====================================================
def extract_main_subject_global(segments):
    full_text = " ".join(segments[:5])
    entities = extract_entities(full_text)
    return entities[0] if entities else "person"

# =====================================================
# STRUCTURED PROMPT
# =====================================================
def build_structured_prompt(segment, main_subject):

    entities = extract_entities(segment)
    secondary = entities[1] if len(entities) > 1 else None

    structure = []

    # 🔥 HARD LOCK
    structure.append(f"(one {main_subject}:1.8) as main subject")

    if secondary and secondary != main_subject:
        structure.append(f"(one {secondary}:1.2) as secondary subject")

    # Spatial constraints
    text = segment.lower()

    if "cage" in text:
        structure.append("main subject is inside the cage, enclosed within bars")

    if "sky" in text:
        structure.append("subject is outdoors under open sky")

    if "water" in text:
        structure.append("subject is interacting with water")

    structure.append("no duplicate subjects")

    return ", ".join(structure) + f". {segment}"

# =====================================================
# IMAGE BLENDING
# =====================================================
def triple_blend(prev, style, identity, a=0.5, b=0.3):
    prev = np.array(prev).astype(np.float32)
    style = np.array(style).astype(np.float32)
    identity = np.array(identity).astype(np.float32)

    blended = a * prev + b * style + (1 - a - b) * identity
    return Image.fromarray(np.uint8(blended))

def blend_images(img1, img2, alpha=0.3):
    img1 = np.array(img1).astype(np.float32)
    img2 = np.array(img2).astype(np.float32)
    blended = alpha * img1 + (1 - alpha) * img2
    return Image.fromarray(np.uint8(blended))

# =====================================================
# VIDEO GENERATION
# =====================================================
def generate_poem_video(poem_text, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    segments = [l.strip() for l in poem_text.split("\n") if l.strip()]
    if len(segments) == 0:
        return

    segments = segments[:12]
    print("Segments:", len(segments))

    # 🔥 GLOBAL SUBJECT
    main_subject = extract_main_subject_global(segments)
    print("Main subject:", main_subject)

    # =====================================================
    # STYLE ANCHOR
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

    style_anchor.save(os.path.join(output_dir, "anchor.png"))

    previous_image = style_anchor
    identity_anchor = None
    generated_paths = []

    # =====================================================
    # SCENE GENERATION
    # =====================================================
    for idx, segment in enumerate(segments):

        print(f"\nScene {idx+1}/{len(segments)}")
        print(segment)

        structured_segment = build_structured_prompt(segment, main_subject)

        prompt = (
            f"{GLOBAL_STYLE}, {BASE_STYLE}, {IDENTITY_LOCK}, "
            f"{structured_segment}, realistic interactions"
        )

        # 🔥 BLENDING STRATEGY
        if identity_anchor is not None:
            input_image = triple_blend(previous_image, style_anchor, identity_anchor)
        else:
            input_image = blend_images(previous_image, style_anchor)

        # 🔥 BETTER STABILITY
        strength = 0.5 if idx == 0 else 0.42

        scene_generator = torch.Generator(device=DEVICE).manual_seed(1000 + idx)

        result = img2img(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=input_image,
            strength=strength,
            guidance_scale=7.5,
            num_inference_steps=40,
            generator=scene_generator
        )

        image = result.images[0]

        # 🔥 SET IDENTITY ANCHOR
        if idx == 0:
            identity_anchor = image

        path = os.path.join(output_dir, f"scene_{idx}.png")
        image.save(path)

        generated_paths.append(path)
        previous_image = image

    # =====================================================
    # VIDEO CREATION
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
    print("====================================")

    try:
        generate_poem_video(poem_text, poem_output_dir)

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    except Exception as e:
        print("Failed for:", title)
        print(e)

print("\nAll poems processed.")