import tempfile
import unittest
from pathlib import Path

from idea.config import ConfigError, load_config
from idea.energy import EnergyError, calculate_energy
from idea.hashing import stable_hash
from idea.pipeline import IdeaPipeline, count_dna_residues
from idea.preprocess import prepare_data_dir
from idea.strategy2 import fit_ridge_gamma


def write_min_pdb(path: Path):
    path.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  O5'  DA B   1       1.000   0.000   0.000  1.00  0.00           O\n"
        "ATOM      3  O5'  DT B   2       2.000   0.000   0.000  1.00  0.00           O\n"
        "END\n"
    )


class IdeaCliCoreTests(unittest.TestCase):
    def test_energy_rejects_dimension_mismatch(self):
        gamma = [1.0, 2.0, 3.0]
        phi = [[1.0, 2.0]]
        with self.assertRaises(EnergyError):
            calculate_energy(gamma, phi)

    def test_stable_hash_changes_when_input_changes(self):
        first = stable_hash({"a": 1})
        second = stable_hash({"a": 2})
        self.assertNotEqual(first, second)
        self.assertEqual(first, stable_hash({"a": 1}))

    def test_config_validation_rejects_bad_dna_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdb = root / "x_modified.pdb"
            seq = root / "seq.txt"
            write_min_pdb(pdb)
            seq.write_text("A\n")
            cfg = root / "bad.yaml"
            cfg.write_text(
                "run_id: bad\n"
                "defaults:\n"
                "  dna_mode: invalid\n"
                "training_sets:\n"
                "  t:\n"
                "    pdbs:\n"
                "      - id: x\n"
                "        path: x_modified.pdb\n"
                "testing_sets:\n"
                "  s:\n"
                "    pdb:\n"
                "      id: x\n"
                "      path: x_modified.pdb\n"
                "    sequences: seq.txt\n"
            )
            with self.assertRaises(ConfigError):
                load_config(cfg)


    def test_simple_config_auto_generates_run_id_and_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_dir = root / "training" / "PDBs"
            test_dir = root / "testing" / "PDBs"
            seq_dir = root / "testing" / "sequences"
            train_dir.mkdir(parents=True)
            test_dir.mkdir(parents=True)
            seq_dir.mkdir(parents=True)
            write_min_pdb(train_dir / "x_modified.pdb")
            write_min_pdb(test_dir / "x_modified.pdb")
            (seq_dir / "dna_half.seq").write_text("A\n")
            cfg = root / "simple.yaml"
            cfg.write_text(
                "defaults:\n"
                "  max_testing_decoys: 10\n"
                "training:\n"
                "  pdbs:\n"
                "    - x\n"
                "testing:\n"
                "  - id: x\n"
                "    dna_mode: ds\n"
            )
            config = load_config(cfg)
            self.assertTrue(config["_run_id_auto_generated"])
            self.assertIn("train_x", config["training_sets"])
            self.assertIn("test_x", config["testing_sets"])
            self.assertEqual(config["jobs"][0]["name"], "train_x__test_x")
            dry_run = IdeaPipeline(config, repo_root=Path.cwd()).dry_run()
            job = dry_run["jobs"][0]
            self.assertIn("runs/cache/training/train_x__", job["training_artifact"])
            self.assertIn("runs/cache/testing/test_x__", job["testing_artifact"])
            self.assertIn("runs/cache/energy/train_x__test_x__", job["energy_artifact"])


    def test_prepare_data_dir_writes_training_and_testing_modified_pdbs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            raw = data / "raw1.pdb"
            raw.write_text(
                "ATOM      1  CA  ALA X   1       0.000   0.000   0.000  1.00  0.00           C\n"
                "ATOM      2  H   ALA X   1       0.000   0.000   0.000  1.00  0.00           H\n"
                "HETATM    3  O   HOH W   1       0.000   0.000   0.000  1.00  0.00           O\n"
                "ATOM      4  O5'  DA Y   1       1.000   0.000   0.000  1.00  0.00           O\n"
                "END\n"
            )
            prepared = prepare_data_dir(data, root)
            self.assertEqual(len(prepared), 1)
            train_out = root / "training" / "PDBs" / "raw1_modified.pdb"
            test_out = root / "testing" / "PDBs" / "raw1_modified.pdb"
            self.assertTrue(train_out.exists())
            self.assertTrue(test_out.exists())
            out_text = train_out.read_text()
            self.assertIn("ALA A", out_text)
            self.assertIn("DA B", out_text)
            self.assertNotIn("HOH", out_text)
            self.assertNotIn(" H   ALA", out_text)

    def test_strategy2_config_can_skip_training_requirement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_dir = root / "testing" / "PDBs"
            seq_dir = root / "testing" / "sequences"
            strategy_dir = root / "data" / "strategy2" / "x"
            test_dir.mkdir(parents=True)
            seq_dir.mkdir(parents=True)
            strategy_dir.mkdir(parents=True)
            write_min_pdb(test_dir / "x_modified.pdb")
            (seq_dir / "dna_half.seq").write_text("A\n")
            (strategy_dir / "dna_half.seq").write_text("A\n")
            (strategy_dir / "exp.txt").write_text("1.5\n")
            cfg = root / "strategy2.yaml"
            cfg.write_text(
                "run_id: s2\n"
                "testing:\n"
                "  - id: x\n"
                "    dna_mode: ds\n"
                "strategy2:\n"
                "  runs:\n"
                "    ridge_x:\n"
                "      template_testing: test_x\n"
                "      sequences: data/strategy2/x/dna_half.seq\n"
                "      exp: data/strategy2/x/exp.txt\n"
            )
            with self.assertRaises(ConfigError):
                load_config(cfg)
            config = load_config(cfg, require_training=False)
            self.assertIn("test_x", config["testing_sets"])
            self.assertEqual(config["training_sets"], {})

    def test_strategy2_ridge_fit_returns_phi_sized_gamma(self):
        phi = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        exp = [1.0, 2.0, 3.0]
        fit = fit_ridge_gamma(
            phi,
            exp,
            {
                "alpha": 1.0,
                "alpha_grid": {"min_exp": -4.0, "max_exp": 2.0, "num": 10},
                "fit_intercept": False,
                "target_sign": "negate",
            },
        )
        self.assertEqual(fit["gamma"].shape[0], 2)
        self.assertEqual(fit["intercept"], 0.0)

    def test_testing_hash_distinguishes_ss_and_ds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdb = root / "x_modified.pdb"
            seq = root / "seq.txt"
            write_min_pdb(pdb)
            seq.write_text("AA\n")
            cfg = root / "ok.yaml"
            cfg.write_text(
                "run_id: ok\n"
                "training_sets:\n"
                "  t:\n"
                "    pdbs:\n"
                "      - id: x\n"
                "        path: x_modified.pdb\n"
                "testing_sets:\n"
                "  s:\n"
                "    pdb:\n"
                "      id: x\n"
                "      path: x_modified.pdb\n"
                "    sequences: seq.txt\n"
                "    dna_mode: ss\n"
            )
            config = load_config(cfg)
            pipe = IdeaPipeline(config, repo_root=Path.cwd())
            ss_hash = pipe.testing_key("s", config["testing_sets"]["s"])
            config["testing_sets"]["s"]["dna_mode"] = "ds"
            ds_hash = pipe.testing_key("s", config["testing_sets"]["s"])
            self.assertNotEqual(ss_hash, ds_hash)
            self.assertEqual(count_dna_residues(pdb), 2)


if __name__ == "__main__":
    unittest.main()
