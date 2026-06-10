"""GEMS command-line interface (fire). [CONCRETE wiring]

Subcommands (some bodies are stubs that raise NotImplementedError with guidance):

    gems build-vocab  --mds_dir data/mds_csv               --out data/processed/vocab.json
    gems build-corpus --pks_dir data/walking_calibrated_pks --out data/processed/dev.h5
    gems pretrain     --config configs/experiment/gems_pretrain.yaml
    gems finetune     --config configs/finetune/sample_classification.yaml
    gems embed        --pks path/to.pks                    --out emb.npy
    gems smoke                                              # CPU end-to-end skeleton check
"""

from __future__ import annotations

import glob
from pathlib import Path


class CLI:
    """GEMS commands. Each method is a thin entry into the package."""

    def build_vocab(self, mds_dir: str, out: str, seed: bool = True, top_k: int = 128):
        """Build the building-block Δm vocabulary from the corpus MDS CSVs. [STUB body]"""
        from gems.vocab.vocabulary import build_vocabulary_from_corpus
        vocab = build_vocabulary_from_corpus(mds_dir, seed=seed, top_k=top_k)
        vocab.to_json(out)
        return out

    def build_corpus(self, pks_dir: str, out: str, limit: int | None = None):
        """Build the HDF5 spectrum corpus from a directory of .pks files. [STUB body]"""
        from gems.data.ms_data import MSData
        MSData.from_pks_dir(pks_dir, out, limit=limit)
        return out

    def pretrain(self, config: str, **overrides):
        """Run self-supervised pretraining. [STUB body]"""
        from gems.training.train import pretrain
        dotlist = [f"{k}={v}" for k, v in overrides.items()]
        return pretrain(config, overrides=dotlist or None)

    def finetune(self, config: str, **overrides):
        """Run supervised fine-tuning. [STUB body]"""
        from gems.training.train import finetune
        dotlist = [f"{k}={v}" for k, v in overrides.items()]
        return finetune(config, overrides=dotlist or None)

    def embed(self, pks: str, out: str, model: str = "gems_dev"):
        """Embed a .pks spectrum with a pretrained model. [STUB body]"""
        from gems.api import embed_spectrum
        return embed_spectrum(pks, model=model)

    def smoke(self, pks_dir: str = "data/walking_calibrated_pks"):
        """End-to-end skeleton check on CPU. [CONCRETE up to the first stub]

        Exercises the concrete pyc2mc-backed path (read a real .pks → canonical record → seed
        vocabulary), then steps into the model pipeline, which raises NotImplementedError until the
        stubs are filled in (per the build order). Prints exactly where the skeleton stops.
        """
        from gems.data.peaklist import load_record
        from gems.vocab.vocabulary import DeltaVocabulary

        files = sorted(glob.glob(str(Path(pks_dir) / "*.pks")))
        if not files:
            raise FileNotFoundError(f"No .pks files under {pks_dir!r}")

        rec = load_record(files[0])
        print(f"[smoke] read {Path(files[0]).name}: {len(rec)} peaks, ion_mode={rec.ion_mode.value}")

        vocab = DeltaVocabulary.from_seeds(include_c13=True)
        print(f"[smoke] seed vocabulary: {len(vocab)} blocks -> {vocab.names}")

        print("[smoke] concrete data + vocab path OK. Next steps are stubs (see build order):")
        print("        Δm graph → peak_tokenizer → transformer → denoising objective → GEMS.training_step.")
        print("[smoke] To complete the smoke test, fill in those stubs and wire GEMS + datamodule.")
        return "ok-concrete-path"


def main():
    """Console entry point (`gems`)."""
    import fire
    fire.Fire(CLI)


if __name__ == "__main__":
    main()
