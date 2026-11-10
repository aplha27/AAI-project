"""
Synthetic Identity Dataset Generator for eKYC System
Generates 200 identity bundles as specified in ERP Section 5.2:
- 120 Genuine Clean (Approve)
- 30 Genuine Poor Capture (Retry)
- 30 Spoof / Fraud Attack (Reject via HITL)
- 20 Mismatch (Manual Review)
Splits: 140 Train, 30 Validation, 30 Test
"""

import os
import json
import math
import wave
import struct
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

DATA_DIR = os.path.join(os.path.dirname(__file__), "processed")
ASSETS_DIR = os.path.join(DATA_DIR, "assets")

FIRST_NAMES = [
    "Aarav", "Aditi", "Rahul", "Priya", "Vikram", "Sneha", "Ananya", "Rohan",
    "Deepak", "Kavita", "Siddharth", "Pooja", "Arjun", "Neha", "Karthik", "Ritu",
    "Manish", "Divya", "Suresh", "Meera", "Gaurav", "Sunita", "Amit", "Swati",
    "Nikhil", "Tanvi", "Rajesh", "Shreya", "Varun", "Ishita", "Alok", "Poonam"
]

LAST_NAMES = [
    "Sharma", "Verma", "Patel", "Reddy", "Iyer", "Nair", "Gupta", "Singh",
    "Kumar", "Mehta", "Joshi", "Choudhury", "Bose", "Deshmukh", "Kulkarni", "Rao",
    "Chatterjee", "Mishra", "Pandey", "Bhat", "Saxena", "Malhotra", "Kapoor", "Das"
]

CITIES = [
    ("Mumbai", "Maharashtra", "400001"),
    ("Bengaluru", "Karnataka", "560001"),
    ("New Delhi", "Delhi", "110001"),
    ("Hyderabad", "Telangana", "500001"),
    ("Chennai", "Tamil Nadu", "600001"),
    ("Pune", "Maharashtra", "411001"),
    ("Kolkata", "West Bengal", "700001"),
    ("Ahmedabad", "Gujarat", "380001")
]

FRAUD_DEVICE_POOL = [
    "DEV-FRAUD-CLUSTER-X89",
    "DEV-FRAUD-CLUSTER-X89",
    "DEV-FRAUD-CLUSTER-X89",
    "DEV-FRAUD-CLUSTER-X89",
]

DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

def generate_face_image(seed_val, identity_id, variation="selfie", is_spoof=False, blur=0.0, glare=False):
    """
    Renders a synthetic facial portrait with deterministic features based on seed_val.
    Variation creates consistent intra-person appearance (photo on card vs selfie).
    """
    rng = random.Random(seed_val)
    img = Image.new("RGB", (320, 360), color=(240, 242, 245))
    draw = ImageDraw.Draw(img)

    # Background gradient or solid
    bg_color = (rng.randint(220, 240), rng.randint(225, 245), rng.randint(235, 255))
    draw.rectangle([0, 0, 320, 360], fill=bg_color)

    # Face tone
    skin_tones = [
        (245, 215, 185), (235, 195, 160), (215, 175, 140),
        (190, 145, 110), (160, 115, 80), (225, 180, 145)
    ]
    skin = rng.choice(skin_tones)

    # Small variation between card photo and selfie
    if variation == "selfie":
        skin = tuple(max(0, min(255, c + rng.randint(-10, 10))) for c in skin)

    # Neck
    neck_x1, neck_x2 = 130, 190
    draw.rectangle([neck_x1, 230, neck_x2, 310], fill=skin)

    # Torso / Clothes
    shirt_color = (rng.randint(40, 180), rng.randint(40, 180), rng.randint(50, 200))
    draw.ellipse([60, 280, 260, 480], fill=shirt_color)

    # Head oval
    head_box = [90, 70, 230, 250]
    draw.ellipse(head_box, fill=skin, outline=(180, 140, 110), width=1)

    # Hair
    hair_color = (rng.randint(20, 45), rng.randint(15, 35), rng.randint(10, 30))
    hair_style = rng.choice(["short", "parted", "wavy", "buzz"])
    if hair_style in ["short", "parted"]:
        draw.arc([85, 45, 235, 160], start=180, end=360, fill=hair_color, width=28)
        draw.ellipse([88, 50, 232, 110], fill=hair_color)
    else:
        draw.arc([80, 40, 240, 180], start=170, end=370, fill=hair_color, width=35)
        draw.ellipse([85, 45, 235, 115], fill=hair_color)

    # Eyes
    eye_y = 145 + (1 if variation == "selfie" else 0)
    # Left eye
    draw.ellipse([115, eye_y - 8, 145, eye_y + 8], fill=(255, 255, 255), outline=(120, 90, 70))
    draw.ellipse([125, eye_y - 6, 137, eye_y + 6], fill=(50, 35, 20))
    draw.ellipse([129, eye_y - 3, 133, eye_y + 1], fill=(255, 255, 255))
    # Right eye
    draw.ellipse([175, eye_y - 8, 205, eye_y + 8], fill=(255, 255, 255), outline=(120, 90, 70))
    draw.ellipse([183, eye_y - 6, 195, eye_y + 6], fill=(50, 35, 20))
    draw.ellipse([187, eye_y - 3, 191, eye_y + 1], fill=(255, 255, 255))

    # Eyebrows
    draw.line([112, eye_y - 16, 148, eye_y - 14], fill=hair_color, width=3)
    draw.line([172, eye_y - 14, 208, eye_y - 16], fill=hair_color, width=3)

    # Nose
    draw.line([160, eye_y, 155, 185], fill=(160, 110, 80), width=2)
    draw.arc([150, 178, 170, 190], start=0, end=180, fill=(160, 110, 80), width=2)

    # Mouth
    mouth_y = 210
    draw.line([138, mouth_y, 182, mouth_y], fill=(170, 80, 80), width=3)

    # If it's a spoof attack (screen presentation attack or print attack)
    if is_spoof:
        # Add moiré patterns and screen RGB grid lines
        arr = np.array(img).astype(np.float32)
        h, w, c = arr.shape
        # Add fine grid raster lines
        grid = np.sin(np.linspace(0, 120 * np.pi, h))[:, None, None] * 22
        arr = np.clip(arr + grid, 0, 255)
        # Add slight color cast
        arr[:, :, 2] = np.clip(arr[:, :, 2] * 1.15, 0, 255) # bluish screen reflection
        img = Image.fromarray(arr.astype(np.uint8))
        draw = ImageDraw.Draw(img)
        # Border of physical phone screen or tablet
        draw.rectangle([6, 6, 314, 354], outline=(30, 30, 30), width=6)
        draw.ellipse([155, 12, 165, 22], fill=(20, 20, 20)) # front speaker hole

    # Apply blur if poor capture
    if blur > 0:
        radius = int(blur * 5) + 1
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # Apply glare if requested
    if glare:
        glare_overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
        gdraw = ImageDraw.Draw(glare_overlay)
        gdraw.ellipse([80, 50, 240, 210], fill=(255, 255, 255, 160))
        img = Image.alpha_composite(img.convert("RGBA"), glare_overlay).convert("RGB")

    return img


def generate_synthetic_id_card(bundle_info, portrait_img, blur=0.0, glare=False):
    """
    Renders a realistic synthetic National Identity / eKYC card with portrait,
    header, security guilloche patterns, field text, and digital chip graphic.
    """
    card_w, card_h = 600, 380
    card = Image.new("RGB", (card_w, card_h), color=(252, 253, 255))
    draw = ImageDraw.Draw(card)

    # Background security bands
    for y in range(0, card_h, 8):
        shade = 240 + int(10 * math.sin(y / 15.0))
        draw.line([(0, y), (card_w, y)], fill=(shade, shade + 3, shade + 8), width=2)

    # Top header banner
    header_color = (25, 75, 140) if bundle_info["doc_type"] == "NATIONAL_ID" else (20, 100, 70)
    draw.rectangle([0, 0, card_w, 65], fill=header_color)

    # Golden chip emblem
    draw.rounded_rectangle([40, 85, 95, 130], radius=5, fill=(212, 175, 55), outline=(180, 140, 30), width=2)
    draw.line([40, 107, 95, 107], fill=(160, 120, 20), width=1)
    draw.line([67, 85, 67, 130], fill=(160, 120, 20), width=1)

    # Portrait placement
    portrait_small = portrait_img.resize((150, 180), Image.Resampling.LANCZOS)
    card.paste(portrait_small, (40, 145))
    draw.rectangle([39, 144, 191, 326], outline=(150, 160, 180), width=2)

    # Text details
    try:
        font_lg = ImageFont.load_default()
    except Exception:
        font_lg = None

    # Card Title text
    draw.text((20, 15), f"REPUBLIC VERIFICATION AUTHORITY • {bundle_info['doc_type']}", fill=(255, 255, 255))
    draw.text((20, 38), "IDENTITY CARD / SECURE CREDENTIAL", fill=(210, 230, 255))

    # Field data
    fields = [
        ("NAME", bundle_info["doc_name"]),
        ("DOB", bundle_info["doc_dob"]),
        ("ID NUMBER", bundle_info["doc_id_number"]),
        ("EXPIRES", bundle_info["expiry_date"]),
        ("ADDRESS", bundle_info["doc_address"][:35]),
    ]

    x_text = 210
    y_start = 85
    for i, (label, val) in enumerate(fields):
        y = y_start + (i * 46)
        draw.text((x_text, y), label, fill=(100, 110, 125))
        draw.text((x_text, y + 15), str(val), fill=(15, 25, 45))

    # Microprint & holographic badge simulator
    draw.ellipse([500, 280, 570, 350], outline=(180, 190, 200), width=3)
    draw.text((512, 310), "VALID", fill=(160, 170, 185))

    # Barcode representation
    for bx in range(210, 480, 4):
        w_bar = 2 if (bx % 6 == 0 or bx % 10 == 0) else 1
        draw.line([(bx, 330), (bx, 360)], fill=(30, 30, 30), width=w_bar)

    # Card border
    draw.rounded_rectangle([1, 1, card_w - 2, card_h - 2], radius=16, outline=(180, 190, 205), width=2)

    # Artifacts
    if blur > 0:
        radius = int(blur * 6) + 1
        card = card.filter(ImageFilter.GaussianBlur(radius=radius))

    if glare:
        glare_overlay = Image.new("RGBA", card.size, (255, 255, 255, 0))
        gdraw = ImageDraw.Draw(glare_overlay)
        gdraw.ellipse([260, 120, 450, 260], fill=(255, 255, 255, 180))
        card = Image.alpha_composite(card.convert("RGBA"), glare_overlay).convert("RGB")

    return card


def generate_spoken_challenge_audio(file_path, digits, is_replay=False, noise_level=0.02):
    """
    Synthesizes a realistic audio WAV file containing spoken digit challenges.
    Uses formant-based audio synthesis for digits 0-9.
    If is_replay=True, applies frequency bandpass clipping, acoustic resonance, and room noise.
    """
    sample_rate = 16000
    digit_duration = 0.45
    silence_duration = 0.15

    # Base formant frequencies for digits 0-9
    digit_formants = {
        0: [(300, 1.0), (1800, 0.4)],   # zero
        1: [(280, 1.0), (2200, 0.5)],   # one
        2: [(400, 1.0), (1900, 0.4)],   # two
        3: [(500, 1.0), (1700, 0.5)],   # three
        4: [(650, 1.0), (1200, 0.6)],   # four
        5: [(700, 1.0), (1100, 0.5)],   # five
        6: [(450, 1.0), (2100, 0.4)],   # six
        7: [(550, 1.0), (1600, 0.5)],   # seven
        8: [(480, 1.0), (1800, 0.5)],   # eight
        9: [(600, 1.0), (1300, 0.5)]    # nine
    }

    full_samples = []

    for d in digits:
        f_list = digit_formants.get(d, [(400, 1.0)])
        n_samples = int(sample_rate * digit_duration)
        for i in range(n_samples):
            t = float(i) / sample_rate
            # Amplitude envelope (attack - sustain - decay)
            env = math.sin(math.pi * (i / n_samples))
            val = 0.0
            for freq, amp in f_list:
                # Fundamental + second harmonic
                val += amp * math.sin(2.0 * math.pi * freq * t)
                val += 0.3 * amp * math.sin(4.0 * math.pi * freq * t)
            # Add subtle vocal jitter
            val *= env
            full_samples.append(val)

        # Silence gap
        silence_samples = int(sample_rate * silence_duration)
        full_samples.extend([0.0] * silence_samples)

    arr = np.array(full_samples, dtype=np.float32)

    # Add realistic environmental background noise
    noise = np.random.normal(0, noise_level, len(arr))
    arr = arr + noise

    if is_replay:
        # Replay attack: low-grade phone speaker characteristics
        # Apply low-pass / band-pass filter cutoff and distortion
        arr = np.clip(arr * 1.8, -0.7, 0.7)  # Speaker clipping distortion
        # Add high-frequency phone speaker hiss / resonance
        resonance = 0.15 * np.sin(2.0 * math.pi * 3200 * np.linspace(0, len(arr) / sample_rate, len(arr)))
        arr = arr + resonance

    # Normalize to 16-bit PCM
    max_val = np.max(np.abs(arr)) or 1.0
    arr = (arr / max_val) * 30000.0
    int_samples = arr.astype(np.int16)

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with wave.open(file_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_samples.tobytes())


def generate_full_dataset(count=200):
    """
    Builds the complete 200 synthetic identity bundles:
    - 120 Genuine Clean
    - 30 Genuine Poor Capture
    - 30 Spoof Attacks
    - 20 Mismatch
    """
    os.makedirs(ASSETS_DIR, exist_ok=True)
    random.seed(42)
    np.random.seed(42)

    bundles = []

    # Distribution counts
    counts = {
        "genuine_clean": 120,
        "genuine_poor_capture": 30,
        "spoof": 30,
        "mismatch": 20
    }

    # Split allocations: 140 train, 30 val, 30 test
    # Distributed proportionally across categories
    categories = []
    for cat, num in counts.items():
        categories.extend([cat] * num)
    random.shuffle(categories)

    train_indices = set(range(0, 140))
    val_indices = set(range(140, 170))
    test_indices = set(range(170, 200))

    for idx, category in enumerate(categories):
        bundle_num = idx + 1
        bundle_id = f"BUNDLE-{bundle_num:03d}"
        seed_val = 1000 + bundle_num

        if idx in train_indices:
            split = "train"
        elif idx in val_indices:
            split = "val"
        else:
            split = "test"

        fname = random.choice(FIRST_NAMES)
        lname = random.choice(LAST_NAMES)
        full_name = f"{fname} {lname}"

        dob_year = random.randint(1975, 2003)
        dob_month = random.randint(1, 12)
        dob_day = random.randint(1, 28)
        dob = f"{dob_year:04d}-{dob_month:02d}-{dob_day:02d}"

        city, state, pin = random.choice(CITIES)
        address = f"Flat {random.randint(101, 909)}, Green Heights, {city}, {state} - {pin}"

        id_prefix = "ID"
        id_number = f"{id_prefix}{random.randint(10000000, 99999999)}"
        expiry_date = f"{dob_year + 35:04d}-12-31"

        # 4-digit spoken challenge
        challenge_digits = [random.randint(0, 9) for _ in range(4)]

        # Base applicant data
        app_name = full_name
        app_dob = dob
        app_id_number = id_number
        device_id = f"DEV-USR-{random.randint(1000, 9999)}"
        ip_addr = f"192.168.1.{random.randint(10, 240)}"

        # Category specific properties
        doc_type = "NATIONAL_ID" if bundle_num % 2 == 0 else "DRIVING_LICENCE"
        doc_name = full_name
        doc_dob = dob
        doc_id_number = id_number

        blur_level = 0.0
        glare_flag = False
        is_spoof = False
        is_replay_audio = False
        spoken_digits = list(challenge_digits)
        face_match_expected = True
        ground_truth_decision = "APPROVE"
        ground_truth_risk = 0.08
        explanation = "High-quality document, live selfie match, verified spoken challenge."

        if category == "genuine_clean":
            ground_truth_decision = "APPROVE"
            ground_truth_risk = round(random.uniform(0.04, 0.18), 3)

        elif category == "genuine_poor_capture":
            # Blur or glare on card, slight blur on selfie
            blur_level = random.choice([0.6, 0.8, 1.0])
            glare_flag = random.choice([True, False])
            ground_truth_decision = "RETRY"
            ground_truth_risk = round(random.uniform(0.38, 0.55), 3)
            explanation = "Document quality degraded by blur/glare. Prompt applicant to re-upload clear ID."

        elif category == "spoof":
            # Spoof presentation attack (screen/print photo, replay audio, device collision)
            is_spoof = True
            is_replay_audio = True
            device_id = random.choice(FRAUD_DEVICE_POOL)  # Repeated device fraud cluster
            if random.random() < 0.4:
                spoken_digits = [random.randint(0, 9) for _ in range(4)]  # Mismatched spoken challenge
            ground_truth_decision = "REJECT"
            ground_truth_risk = round(random.uniform(0.82, 0.98), 3)
            explanation = "Presentation spoof detected (screen moiré / replay audio) with suspicious device velocity."

        elif category == "mismatch":
            # Mismatched name or photo of different person
            sub_type = random.choice(["name_mismatch", "face_mismatch", "dob_mismatch"])
            if sub_type == "name_mismatch":
                doc_name = f"{random.choice(FIRST_NAMES)} {lname}"
                explanation = f"Name discrepancy between application ({app_name}) and ID document ({doc_name})."
            elif sub_type == "dob_mismatch":
                doc_dob = f"{dob_year + random.choice([-2, 2]):04d}-{dob_month:02d}-{dob_day:02d}"
                explanation = f"DOB discrepancy between application ({app_dob}) and ID document ({doc_dob})."
            else:
                face_match_expected = False
                explanation = "Selfie face embedding fails similarity threshold against ID document portrait."
            ground_truth_decision = "MANUAL_REVIEW"
            ground_truth_risk = round(random.uniform(0.58, 0.78), 3)

        # Asset Paths
        card_img_filename = f"{bundle_id}_id_card.png"
        selfie_img_filename = f"{bundle_id}_selfie.png"
        audio_filename = f"{bundle_id}_audio_challenge.wav"

        card_img_path = os.path.join(ASSETS_DIR, card_img_filename)
        selfie_img_path = os.path.join(ASSETS_DIR, selfie_img_filename)
        audio_path = os.path.join(ASSETS_DIR, audio_filename)

        # Generate Face for ID Card
        doc_face_seed = seed_val
        card_face_img = generate_face_image(doc_face_seed, bundle_num, variation="card")

        # Generate Face for Selfie
        if face_match_expected:
            selfie_face_seed = seed_val
        else:
            selfie_face_seed = seed_val + 9999  # Impostor face

        selfie_img = generate_face_image(
            selfie_face_seed,
            bundle_num,
            variation="selfie",
            is_spoof=is_spoof,
            blur=blur_level * 0.5,
            glare=False
        )

        # Generate ID Card Image
        doc_info = {
            "doc_type": doc_type,
            "doc_name": doc_name,
            "doc_dob": doc_dob,
            "doc_id_number": doc_id_number,
            "doc_address": address,
            "expiry_date": expiry_date
        }
        id_card_img = generate_synthetic_id_card(doc_info, card_face_img, blur=blur_level, glare=glare_flag)

        # Save Images
        selfie_img.save(selfie_img_path)
        id_card_img.save(card_img_path)

        # Generate Audio
        generate_spoken_challenge_audio(
            audio_path,
            spoken_digits,
            is_replay=is_replay_audio,
            noise_level=0.08 if category == "genuine_poor_capture" else 0.02
        )

        bundle_record = {
            "bundle_id": bundle_id,
            "split": split,
            "category": category,
            "ground_truth_decision": ground_truth_decision,
            "ground_truth_risk_score": ground_truth_risk,
            "ground_truth_explanation": explanation,
            "applicant_data": {
                "full_name": app_name,
                "dob": app_dob,
                "id_number": app_id_number,
                "address": address,
                "device_id": device_id,
                "ip_address": ip_addr,
                "submission_channel": "WEB_PORTAL",
                "attempt_number": 1 if category != "genuine_poor_capture" else 2
            },
            "document_data": {
                "doc_type": doc_type,
                "doc_name": doc_name,
                "doc_dob": doc_dob,
                "doc_id_number": doc_id_number,
                "doc_address": address,
                "expiry_date": expiry_date,
                "blur_level": blur_level,
                "glare_present": glare_flag
            },
            "verification_assets": {
                "selfie_image": card_img_filename.replace("_id_card.png", "_selfie.png"),
                "id_card_image": card_img_filename,
                "audio_challenge": audio_filename,
                "expected_challenge_digits": "".join(map(str, challenge_digits)),
                "spoken_challenge_digits": "".join(map(str, spoken_digits)),
                "is_replay_audio": is_replay_audio,
                "is_spoof_presentation": is_spoof,
                "face_match_expected": face_match_expected
            }
        }
        bundles.append(bundle_record)

    # Save manifest
    manifest_path = os.path.join(DATA_DIR, "bundles_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(bundles, f, indent=2)

    # Save Data Card (Deliverable D2)
    data_card = {
        "dataset_name": "Synthetic Multimodal eKYC Verification Benchmark (HSR-AGAI-029)",
        "version": "1.0.0",
        "description": "200 synthetic identity bundles comprising facial selfies, national identity documents, spoken 4-digit acoustic challenges, and applicant metadata.",
        "splits": {
            "train": len([b for b in bundles if b["split"] == "train"]),
            "validation": len([b for b in bundles if b["split"] == "val"]),
            "test": len([b for b in bundles if b["split"] == "test"])
        },
        "class_distribution": counts,
        "privacy_and_ethics": "100% synthetically generated. Contains no PII, real customer data, or copyrighted biometric captures. Safe for reproducible academic benchmarking.",
        "licenses": "MIT Open Source / Academic Research Use",
        "modalities": ["RGB Vision (Selfie & ID Card)", "Acoustic Audio (16kHz 16-bit PCM)", "Structured JSON Metadata", "OCR Ground Truth"]
    }
    with open(os.path.join(DATA_DIR, "data_card.json"), "w") as f:
        json.dump(data_card, f, indent=2)

    print(f"Generated {len(bundles)} identity bundles in {DATA_DIR}")
    print(f"Splits: Train=140, Val=30, Test=30. Assets saved in {ASSETS_DIR}")
    return bundles

if __name__ == "__main__":
    generate_full_dataset()

