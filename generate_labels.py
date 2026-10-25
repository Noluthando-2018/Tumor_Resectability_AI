"""
generate_labels.py

Generates binary resectability labels from the real UCSF-PDGM metadata file.

Label mapping:
    1 = Resectable     → GTR (Gross Total Resection)
    0 = Not resectable → STR (Subtotal Resection)
    EXCLUDED           → Biopsy, missing EOR

Place UCSF-PDGM-metadata.csv in the same folder as this script.

USAGE:
    python generate_labels.py
"""

import os
import csv
import pandas as pd

METADATA_FILE = "UCSF-PDGM-metadata.csv"
DATA_DIR      = os.path.join("Data", "step7_normalized_copy")
OUTPUT_CSV    = "labels.csv"


def generate():
    if not os.path.isfile(METADATA_FILE):
        raise FileNotFoundError(
            f"'{METADATA_FILE}' not found.\n"
            f"Place it in: {os.path.abspath('.')}"
        )

    try:
        df = pd.read_excel(METADATA_FILE, engine="openpyxl")
    except Exception:
        df = pd.read_csv(METADATA_FILE, encoding="latin1")

    print(f"Loaded metadata: {len(df)} patients")
    print(f"EOR value counts:\n{df['EOR'].value_counts().to_string()}\n")

    df = df[df["EOR"].isin(["GTR", "STR"])].copy()
    print(f"After excluding biopsy: {len(df)} patients (GTR + STR only)\n")

    label_lookup = {}
    for _, row in df.iterrows():
        raw_id = str(row["ID"]).strip()
        eor    = str(row["EOR"]).strip()
        label  = 1 if eor == "GTR" else 0
        parts  = raw_id.split("-")
        if len(parts) == 3:
            normalised = f"{parts[0]}-{parts[1]}-{parts[2].zfill(4)}"
        else:
            normalised = raw_id
        label_lookup[normalised] = label
        label_lookup[raw_id]     = label

    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    patients = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ])

    rows    = []
    skipped = []

    for folder_name in patients:
        base_id = folder_name.replace("_nifti", "")
        if base_id in label_lookup:
            label = label_lookup[base_id]
        else:
            parts    = base_id.split("-")
            short_id = f"{parts[0]}-{parts[1]}-{str(int(parts[2]))}" \
                       if len(parts) == 3 else base_id
            if short_id in label_lookup:
                label = label_lookup[short_id]
            else:
                skipped.append(folder_name)
                continue
        rows.append({"patient_id": folder_name, "label": label})

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "label"])
        writer.writeheader()
        writer.writerows(rows)

    gtr  = sum(r["label"] == 1 for r in rows)
    str_ = sum(r["label"] == 0 for r in rows)

    print(f"{'='*50}")
    print(f"Saved {len(rows)} rows to '{OUTPUT_CSV}'")
    print(f"  GTR → Resectable (1)     : {gtr}  ({gtr/len(rows)*100:.1f}%)")
    print(f"  STR → Not resectable (0) : {str_}  ({str_/len(rows)*100:.1f}%)")
    print(f"  Excluded (biopsy/missing): {len(skipped)}")
    print(f"\nClass balance: {'Good ✓' if 0.35 < gtr/len(rows) < 0.65 else 'Imbalanced'}")


if __name__ == "__main__":
    generate()
