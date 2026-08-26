import argparse
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from .utils import load_systems


class ESM2Extractor:
    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_sequence(self, sequence: str) -> torch.Tensor:
        token_ids = self.tokenizer(sequence, return_tensors="pt")["input_ids"].to(self.device)
        outputs = self.model(token_ids).last_hidden_state
        return outputs[0, 1:-1, :].detach().cpu()


def save_plm_for_system(
        system: Dict, 
        extractor: ESM2Extractor, 
        overwrite: bool = False
        ):
    system_dir = Path(system["workdir"])
    protein_pt = system_dir / "protein.pt"
    peptide_pt = system_dir / "peptide.pt"

    if overwrite or not protein_pt.exists():
        protein_emb = extractor.encode_sequence(system["protein_sequence"])
        torch.save(protein_emb, protein_pt)

    if overwrite or not peptide_pt.exists():
        peptide_emb = extractor.encode_sequence(system["peptide_sequence"])
        torch.save(peptide_emb, peptide_pt)


def main():
    parser = argparse.ArgumentParser(description="Generate protein.pt and peptide.pt from systems manifest")
    parser.add_argument("--systems_path", type=str, required=True)
    parser.add_argument("--esm_model", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    systems = load_systems(args.systems_path)
    extractor = ESM2Extractor(model_name=args.esm_model, device=args.device)

    for system in tqdm(systems, desc="Generating PLM features"):
        save_plm_for_system(system, extractor, overwrite=args.overwrite)

    print("Done.")


if __name__ == "__main__":
    main()
