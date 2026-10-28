"""
YOLOE (find + crop plants)  ->  BioCLIP 2 (zero-shot, against candidate list)  ->  crop / not crop

-BioCLIP picks from short list of species(crops + common weeds + a few generic labels), so subspecies and
near-neighbor confusion mostly disappears. 

Install (in conda env):
    pip install -U ultralytics open_clip_torch torch pillow
"""
import csv
from pathlib import Path
 
import open_clip
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLOE
 
# ---------------- settings ----------------
IMAGE_PATH = r"C:\Users\delan\Downloads\your_image.jpg"
BIOCLIP_MODEL = "hf-hub:imageomics/bioclip-2"   # lighter/faster alternative: "hf-hub:imageomics/bioclip"
DETECT_CONF = 0.15        # YOLOE minimum confidence
CROP_THRESHOLD = 0.50     # summed probability on "crop" candidates -> crop
POSSIBLE_THRESHOLD = 0.25 # summed probability on crop + ambiguous candidates -> possible crop
MIN_TOP_CONF = 0.30       # best match below this (and nothing crop-like) -> uncertain
PAD = 0.15                # grow boxes 15% so crops keep leaf/stem context
 
# ---------------- candidate list (EDIT ME) ----------------
# category: "crop" | "ambiguous" (same species is both crop and weed) | "not crop"
CANDIDATES = (
    # --- crops ---
    [(n, "crop") for n in [
        "Zea mays", "Glycine max", "Solanum lycopersicum", "Solanum tuberosum",
        "Solanum melongena", "Capsicum annuum", "Cucurbita pepo", "Cucurbita maxima",
        "Cucumis sativus", "Brassica oleracea", "Allium cepa", "Allium sativum",
        "Lactuca sativa", "Spinacia oleracea", "Beta vulgaris", "Fragaria × ananassa",
        "Malus domestica", "Vaccinium corymbosum", "Phaseolus vulgaris", "Pisum sativum",
        "Ocimum basilicum", "Petroselinum crispum", "Coriandrum sativum", "Chlorophytum comosum"
    ]]
    # --- same species grows as crop AND weed/escape ---
    + [(n, "ambiguous") for n in [
        "Daucus carota",      # carrot vs. Queen Anne's lace
        "Brassica rapa",      # turnip/bok choy vs. field mustard
        "Mentha spicata",     # spearmint crop vs. escaped
    ]]
    # --- common Northeast weeds and look-alikes (the "negatives") ---
    + [(n, "not crop") for n in [
        "Taraxacum officinale", "Amaranthus retroflexus", "Chenopodium album",
        "Ambrosia artemisiifolia", "Cirsium arvense", "Plantago major", "Trifolium repens",
        "Portulaca oleracea", "Digitaria sanguinalis", "Echinochloa crus-galli",
        "Setaria viridis", "Elymus repens", "Poa annua", "Abutilon theophrasti",
        "Solanum carolinense", "Solanum nigrum", "Rumex crispus", "Convolvulus arvensis",
        "Oxalis stricta", "Stellaria media", "Capsella bursa-pastoris", "Alliaria petiolata",
        "Artemisia vulgaris", "Achillea millefolium", "Allium vineale", "Lactuca serriola",
        "Galinsoga quadriradiata", "Persicaria maculosa", "Erigeron canadensis",
        "Rosa multiflora", "Toxicodendron radicans", "Parthenocissus quinquefolia",
        "Acer rubrum", "Quercus rubra", "Philodendron hederaceum",
        # generic catch-alls so odd plants have somewhere to land
        "lawn grass", "tree seedling", "fern", "moss",
    ]]
)
LABELS = [n for n, _ in CANDIDATES]
CATEGORIES = [c for _, c in CANDIDATES]
 
# ---------------- load models once ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
 
detector = YOLOE("yoloe-11l-seg.pt")
detector.set_classes(["plant"], detector.get_text_pe(["plant"]))
 
clip_model, _, preprocess = open_clip.create_model_and_transforms(BIOCLIP_MODEL)
tokenizer = open_clip.get_tokenizer(BIOCLIP_MODEL)
clip_model = clip_model.to(device).eval()
 
with torch.no_grad():
    tokens = tokenizer([f"a photo of {n}" for n in LABELS]).to(device)
    text_features = clip_model.encode_text(tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)  # computed once, reused for every crop
 
 
def classify(crop: Image.Image, k: int = 5):
    """Return probabilities over CANDIDATES for one crop."""
    with torch.no_grad():
        img = preprocess(crop).unsqueeze(0).to(device)
        feat = clip_model.encode_image(img)
        feat /= feat.norm(dim=-1, keepdim=True)
        logits = clip_model.logit_scale.exp() * feat @ text_features.T
        return logits.softmax(dim=-1)[0].cpu()
 
 
# ---------------- run ----------------
image = Image.open(IMAGE_PATH).convert("RGB")
W, H = image.size
result = detector.predict(image, conf=DETECT_CONF, verbose=False)[0]
 
out_dir = Path(IMAGE_PATH).with_suffix("")
out_dir.mkdir(exist_ok=True)
annotated = image.copy()
draw = ImageDraw.Draw(annotated)
COLORS = {"crop": "lime", "possible crop": "orange", "not crop": "red", "uncertain": "yellow"}
# Text shown on the annotated image: always reads CROP or NOT CROP, with a qualifier when unsure
LABEL_TEXT = {
    "crop": "CROP",
    "possible crop": "CROP (possible)",
    "not crop": "NOT CROP",
    "uncertain": "NOT CROP (uncertain)",
}
 
 
def load_font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):  # Windows, then Linux fallback
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()
 
 
font = load_font(max(14, W // 50))   # label size scales with image width
line_w = max(3, W // 400)
 
 
def draw_label(box, plant_id, category):
    color = COLORS[category]
    draw.rectangle(box, outline=color, width=line_w)
    text = f"#{plant_id} {LABEL_TEXT[category]}"
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    tw, th = r - l, b - t
    pad = 4
    x, y = box[0], box[1] - th - 2 * pad          # put label above the box...
    if y < 0:                                      # ...or inside it if there's no room
        y = box[1]
    draw.rectangle((x, y, x + tw + 2 * pad, y + th + 2 * pad), fill=color)
    draw.text((x + pad - l, y + pad - t), text, fill="black", font=font)
 
plants = {}
for plant_id, xyxy in enumerate(result.boxes.xyxy.tolist(), start=1):
    x1, y1, x2, y2 = xyxy
    pw, ph = (x2 - x1) * PAD, (y2 - y1) * PAD
    box = (max(0, x1 - pw), max(0, y1 - ph), min(W, x2 + pw), min(H, y2 + ph))
    crop = image.crop(tuple(int(v) for v in box))
    crop.save(out_dir / f"plant_{plant_id}.jpg")
 
    probs = classify(crop)
    top_vals, top_idx = probs.topk(5)
    top5 = [(LABELS[i], round(float(v), 3)) for v, i in zip(top_vals, top_idx)]
 
    crop_prob = sum(float(p) for p, c in zip(probs, CATEGORIES) if c == "crop")
    ambig_prob = sum(float(p) for p, c in zip(probs, CATEGORIES) if c == "ambiguous")
    top_prob = top5[0][1]
 
    if crop_prob >= CROP_THRESHOLD:
        category = "crop"
    elif crop_prob + ambig_prob >= POSSIBLE_THRESHOLD:
        category = "possible crop"
    elif top_prob < MIN_TOP_CONF:
        category = "uncertain"
    else:
        category = "not crop"
 
    plants[plant_id] = {
        "top_species": top5[0][0],
        "top_confidence": top_prob,
        "top5": top5,
        "crop_prob": crop_prob,
        "ambiguous_prob": ambig_prob,
        "category": category,
        "bbox": box,
        "centroid": ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
    }
 
    draw_label(box, plant_id, category)
 
annotated.save(out_dir / "annotated.jpg")
 
# ---------------- report ----------------
if not plants:
    print("No plants detected. Try lowering DETECT_CONF.")
for pid, p in plants.items():
    print(f"Plant {pid}: {p['category'].upper()}")
    print(f"  Best match: {p['top_species']} ({p['top_confidence']:.2f})")
    print(f"  Crop prob: {p['crop_prob']:.2f} | Ambiguous prob: {p['ambiguous_prob']:.2f}")
    print(f"  Top 5: {p['top5']}")
    print(f"  Centroid: ({p['centroid'][0]:.0f}, {p['centroid'][1]:.0f})\n")
 
with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["plant_id", "category", "best_match", "confidence", "crop_prob", "x1", "y1", "x2", "y2"])
    for pid, p in plants.items():
        w.writerow([pid, p["category"], p["top_species"], round(p["top_confidence"], 3),
                    round(p["crop_prob"], 3), *[round(v) for v in p["bbox"]]])
 
print(f"Crops, annotated image and results.csv saved in: {out_dir}")