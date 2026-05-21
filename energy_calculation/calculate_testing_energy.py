import argparse
from pathlib import Path

from idea.energy import calculate_energy_files


def main():
    parser = argparse.ArgumentParser(description="Calculate IDEA testing energies from explicit gamma and phi files.")
    parser.add_argument("--gamma", default="native_trainSetFiles_phi_pairwise_contact_well-8.0_8.0_0.7_10_gamma_filtered")
    parser.add_argument("--phi", default="phi_pairwise_contact_well_native_decoys_CPLEX_randomization_-8.0_8.0_0.7_10")
    parser.add_argument("--out", default=".", help="Output directory for Energy_mg.txt and manifest.json")
    args = parser.parse_args()

    energy_path = calculate_energy_files(args.gamma, args.phi, args.out)
    print(f"Energy written to: {energy_path}")


if __name__ == "__main__":
    main()
