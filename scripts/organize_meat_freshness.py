
"""

organize_meat_freshness.py

Buckets the Mendeley "Meat Species and Hourly Freshness" dataset's time-point

folders into fresh / half_fresh / spoiled classes, based on elapsed time

since slaughter:

    0 hr, 12 hr   -> fresh

    24 hr         -> half_fresh

    36 hr, 48 hr  -> spoiled

Pools both Beef and Mutton together by default (more data, same freshness

task) — pass --beef-only if you want beef exclusively.

USAGE:

    python organize_meat_freshness.py "<path to 'Original Images' folder>" datasets/freshness

    python organize_meat_freshness.py "<path to 'Original Images' folder>" datasets/freshness --beef-only

"""

import argparse

import re

import shutil

from pathlib import Path

HOUR_TO_CLASS = {

    0: "fresh",

    12: "fresh",

    24: "half_fresh",

    36: "spoiled",

    48: "spoiled",

}

def extract_hour(folder_name: str):

    match = re.match(r"(\d+)\s*hr", folder_name.strip(), flags=re.IGNORECASE)

    return int(match.group(1)) if match else None

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("source", help="Path to the 'Original Images' folder")

    parser.add_argument("dest", help="Path to your datasets/freshness folder")

    parser.add_argument("--beef-only", action="store_true", help="Skip Mutton, use Beef only")

    args = parser.parse_args()

    source_root = Path(args.source).expanduser()

    dest_root = Path(args.dest).expanduser()

    if not source_root.exists():

        print(f"Source folder not found: {source_root}")

        return

    species_dirs = ["Beef"] if args.beef_only else ["Beef", "Mutton"]

    counts = {}

    for species in species_dirs:

        species_path = source_root / species

        if not species_path.exists():

            print(f"Warning: {species_path} not found, skipping")

            continue

        for timepoint_dir in species_path.iterdir():

            if not timepoint_dir.is_dir():

                continue

            hour = extract_hour(timepoint_dir.name)

            if hour is None or hour not in HOUR_TO_CLASS:

                print(f"Warning: couldn't parse hour from '{timepoint_dir.name}', skipping")

                continue

            dest_class = HOUR_TO_CLASS[hour]

            dest_dir = dest_root / dest_class

            dest_dir.mkdir(parents=True, exist_ok=True)

            images = list(timepoint_dir.glob("*.jpg")) + list(timepoint_dir.glob("*.jpeg")) + list(timepoint_dir.glob("*.png")) + list(timepoint_dir.glob("*.JPG")) + list(timepoint_dir.glob("*.PNG"))

            for img_path in images:

                new_name = f"{species.lower()}_{hour}hr_{img_path.name}"

                shutil.copy2(img_path, dest_dir / new_name)

                counts[dest_class] = counts.get(dest_class, 0) + 1

    print("\nOrganized into datasets/freshness/. Images per class:")

    for class_name, count in counts.items():

        print(f"  {class_name}: {count}")

if __name__ == "__main__":

    main()

