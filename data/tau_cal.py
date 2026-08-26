import argparse
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from .label import collect_dataset_tau
from .utils import load_systems, resolve_peptide_path, systems_to_dict, save_tau


def build_tau_samples_from_manifest(
    systems: List[Dict],
    csv_path: str,
) -> List[Dict]:
    df = pd.read_csv(csv_path)
    system_map = systems_to_dict(systems)

    samples = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Collecting tau samples"):
        target = row["TARGET"]
        decoy_id = int(row["DECOY_ID"])

        system = system_map[target]
        peptide_path = resolve_peptide_path(system, decoy_id)

        samples.append({
            "native_peptide_pdb": system["native_fixed"],
            "decoy_peptide_pdb": peptide_path,
        })

    return samples


def main():
    parser = argparse.ArgumentParser(description="Calculate tau from systems manifest and train csv")
    parser.add_argument("--systems_path", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--percentile", type=float, default=75.0)
    args = parser.parse_args()

    systems = load_systems(args.systems_path)
    samples = build_tau_samples_from_manifest(systems, args.csv_path)
    tau_dict = collect_dataset_tau(samples, percentile=args.percentile)

    save_tau(tau_dict, args.save_path)
    print(f"[Saved] tau -> {args.save_path}")
    print(tau_dict)


if __name__ == "__main__":
    main()