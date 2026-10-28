"""
Name each plant, and decide whether it is a crop.

YOLOE finds plants but is bad at saying which - "poppy" scored 0.00 on a
photo of a poppy, where "flower" scored 0.24. So it only ever gets asked
"is there a plant here", and this answers "what is it".

BioCLIP 2 is trained on biological imagery and is asked to pick from a short
list rather than name anything in the world. A short list is what kills the
near-neighbour confusion: a basil plant only has to beat the other few dozen
candidates, not every species there is.

The candidate list and thresholds below come from Delaney's
BioClip_pipeline.py. Her work, moved into the pipeline so one run does
everything.

    pip install open_clip_torch

First use downloads the model (about 1 GB) and is slow. After that it is
cached.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Candidates. category is "crop", "ambiguous" or "not crop".
# ---------------------------------------------------------------------------

CANDIDATES = (
    [(n, "crop") for n in [
        "Zea mays", "Glycine max", "Solanum lycopersicum", "Solanum tuberosum",
        "Solanum melongena", "Capsicum annuum", "Cucurbita pepo",
        "Cucurbita maxima", "Cucumis sativus", "Brassica oleracea",
        "Allium cepa", "Allium sativum", "Lactuca sativa",
        "Spinacia oleracea", "Beta vulgaris", "Fragaria x ananassa",
        "Malus domestica", "Vaccinium corymbosum", "Phaseolus vulgaris",
        "Pisum sativum", "Ocimum basilicum", "Petroselinum crispum",
        "Coriandrum sativum",
    ]]
    # Same species grows as crop AND weed, so it cannot settle it alone.
    + [(n, "ambiguous") for n in [
        "Daucus carota",      # carrot vs Queen Anne's lace
        "Brassica rapa",      # turnip, bok choy vs field mustard
        "Mentha spicata",     # spearmint, crop vs escaped
    ]]
    # Northeast weeds and look-alikes - the negatives.
    + [(n, "not crop") for n in [
        "Taraxacum officinale", "Amaranthus retroflexus", "Chenopodium album",
        "Ambrosia artemisiifolia", "Cirsium arvense", "Plantago major",
        "Trifolium repens", "Portulaca oleracea", "Digitaria sanguinalis",
        "Echinochloa crus-galli", "Setaria viridis", "Elymus repens",
        "Poa annua", "Abutilon theophrasti", "Solanum carolinense",
        "Solanum nigrum", "Rumex crispus", "Convolvulus arvensis",
        "Oxalis stricta", "Stellaria media", "Capsella bursa-pastoris",
        "Alliaria petiolata", "Artemisia vulgaris", "Achillea millefolium",
        "Allium vineale", "Lactuca serriola", "Galinsoga quadriradiata",
        "Persicaria maculosa", "Erigeron canadensis", "Rosa multiflora",
        "Toxicodendron radicans", "Parthenocissus quinquefolia",
        "Acer rubrum", "Quercus rubra",
        # The pipeline treats this one as a negative, not a crop - it is
        # the weed in the indoor test, with the spider plant as the crop.
        "Philodendron hederaceum",
        # Generic catch-alls, so odd plants have somewhere to land rather
        # than being forced into the nearest species.
        "lawn grass", "tree seedling", "fern", "moss",
    ]]
    # Houseplants, for bench testing indoors before there is a real bed.
    #
    # These need no training. BioCLIP already knows these taxa from
    # TreeOfLife-10M - all it needs is permission to choose their names,
    # which is what putting them on this list does.
    #
    # Pothos and monstera are here because both are sold as "philodendron"
    # and the heartleaf kind is near enough identical to pothos that even
    # garden centres mix them up. They count as crop rather than as
    # negatives, so that if one of them wins the vote the plant still
    # lands on the map instead of being discarded as a weed.
    + [(n, "crop") for n in [
        "Chlorophytum comosum",
        "Philodendron bipinnatifidum",
        "Epipremnum aureum",
        "Monstera deliciosa",
    ]]
)

LABELS = [n for n, _ in CANDIDATES]
CATEGORIES = [c for _, c in CANDIDATES]

MODEL = "hf-hub:imageomics/bioclip-2"   # lighter: "hf-hub:imageomics/bioclip"

CROP_THRESHOLD = 0.50       # summed probability on crops -> crop
POSSIBLE_THRESHOLD = 0.25   # crops + ambiguous -> possible crop
MIN_TOP_CONF = 0.30         # best match below this and nothing crop-like
                            # -> uncertain


# Common names, so the menu does not read like a taxonomy exam.
COMMON = {
    "Zea mays": "sweetcorn", "Glycine max": "soybean",
    "Solanum lycopersicum": "tomato", "Solanum tuberosum": "potato",
    "Solanum melongena": "aubergine", "Capsicum annuum": "pepper",
    "Cucurbita pepo": "courgette, summer squash",
    "Cucurbita maxima": "winter squash", "Cucumis sativus": "cucumber",
    "Brassica oleracea": "cabbage, kale, broccoli",
    "Allium cepa": "onion", "Allium sativum": "garlic",
    "Lactuca sativa": "lettuce", "Spinacia oleracea": "spinach",
    "Beta vulgaris": "beetroot, chard", "Fragaria x ananassa": "strawberry",
    "Malus domestica": "apple", "Vaccinium corymbosum": "blueberry",
    "Phaseolus vulgaris": "bean", "Pisum sativum": "pea",
    "Ocimum basilicum": "basil", "Petroselinum crispum": "parsley",
    "Coriandrum sativum": "coriander",
    "Daucus carota": "carrot  (also Queen Anne's lace)",
    "Brassica rapa": "turnip, bok choy  (also field mustard)",
    "Mentha spicata": "spearmint  (also escapes as a weed)",
    "Chlorophytum comosum": "spider plant",
    "Philodendron hederaceum": "heartleaf philodendron",
    "Philodendron bipinnatifidum": "split-leaf philodendron",
    "Epipremnum aureum": "pothos  (sold as philodendron, looks like it)",
    "Monstera deliciosa": "monstera  (sold as split-leaf philodendron)",
}


def crop_choices():
    """The crops and ambiguous species, for a menu. Weeds are not offered."""
    return [(n, c, COMMON.get(n, "")) for n, c in CANDIDATES
            if c in ("crop", "ambiguous")]


def build_candidates(keep=None):
    """
    Narrow the list to the crops actually planted in this bed.

    `keep` is a list of species names; everything else in the crop and
    ambiguous groups is dropped. The weeds always stay, because they are
    what the crops are being told apart FROM - remove those and every
    dandelion has to land on a crop name.

    Narrowing helps. A basil bed classified against 23 crops has 22 ways
    to be confidently wrong; against one crop it has none. The model is
    picking a winner from a list, so a shorter list is a better question.
    """
    if not keep:
        return CANDIDATES

    wanted = {k.strip().lower() for k in keep if k.strip()}
    out = []
    for name, cat in CANDIDATES:
        if cat in ("crop", "ambiguous"):
            if name.lower() in wanted or COMMON.get(name, "").lower() in wanted:
                out.append((name, cat))
        else:
            out.append((name, cat))
    return out


class Classifier:
    """
    Loads BioCLIP once and names crops with it.

        bc = Classifier(candidates=build_candidates(["Ocimum basilicum"]))
        result = bc.classify(pil_image)
        result["category"]  -> crop | possible crop | not crop | uncertain
    """

    def __init__(self, model=MODEL, device=None, candidates=None):
        try:
            import open_clip
            import torch
        except ImportError:
            raise SystemExit(
                "BioCLIP needs open_clip_torch:\n"
                "    python -m pip install open_clip_torch\n"
                "First run also downloads about 1 GB of model.")

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.candidates = candidates or CANDIDATES
        self.labels = [n for n, _ in self.candidates]
        self.cats = [c for _, c in self.candidates]

        self.model, _, self.preprocess = \
            open_clip.create_model_and_transforms(model)
        self.model = self.model.to(self.device).eval()
        tokenizer = open_clip.get_tokenizer(model)

        # The text side never changes, so embed it once and reuse it for
        # every crop. This is most of what makes it fast enough to run on
        # a few dozen plants.
        with torch.no_grad():
            tokens = tokenizer([f"a photo of {n}" for n in self.labels]
                               ).to(self.device)
            feats = self.model.encode_text(tokens)
            self.text_features = feats / feats.norm(dim=-1, keepdim=True)

    def probabilities(self, crop):
        torch = self.torch
        with torch.no_grad():
            img = self.preprocess(crop).unsqueeze(0).to(self.device)
            feat = self.model.encode_image(img)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            logits = self.model.logit_scale.exp() * feat @ self.text_features.T
            return logits.softmax(dim=-1)[0].cpu()

    def classify(self, crop, top_k=5):
        """
        What is this, and is it a crop?

        Decides on SUMMED probability by category rather than the single
        best match. A plant split evenly across three crop species is still
        clearly a crop, even though no one name wins - and picking the top
        label alone would throw that away.
        """
        probs = self.probabilities(crop)
        vals, idx = probs.topk(min(top_k, len(self.labels)))
        top = [(self.labels[i], round(float(v), 3)) for v, i in zip(vals, idx)]

        crop_p = sum(float(p) for p, c in zip(probs, self.cats) if c == "crop")
        ambig_p = sum(float(p) for p, c in zip(probs, self.cats)
                      if c == "ambiguous")
        top_p = top[0][1]

        if crop_p >= CROP_THRESHOLD:
            category = "crop"
        elif crop_p + ambig_p >= POSSIBLE_THRESHOLD:
            category = "possible crop"
        elif top_p < MIN_TOP_CONF:
            category = "uncertain"
        else:
            category = "not crop"

        return {
            "species": top[0][0],
            "confidence": top_p,
            "top": top,
            "crop_prob": round(crop_p, 3),
            "ambiguous_prob": round(ambig_p, 3),
            "category": category,
        }


def crop_image(image, box, margin=0.15):
    """Cut a detection out of the photo, with a little context round it."""
    x1, y1, x2, y2 = box
    w, h = image.size
    px, py = (x2 - x1) * margin, (y2 - y1) * margin
    return image.crop((max(0, int(x1 - px)), max(0, int(y1 - py)),
                       min(w, int(x2 + px)), min(h, int(y2 + py))))
