# Foundation models for atomistic simulations — hands-on

A ~2 hour hands-on session built around [PET-MAD-XS](https://www.nature.com/articles/s41467-025-65662-7),
a foundation (universal) interatomic potential trained with
[`metatrain`](https://github.com/metatensor/metatrain), used through
[`metatomic`](https://docs.metatensor.org/metatomic/latest/index.html) in ASE and LAMMPS.

Two independent notebooks, either of which can be run on its own:

* [`ethanol-finetune/ethanol_finetune.ipynb`](ethanol-finetune/ethanol_finetune.ipynb) —
  **fine-tuning**. Use PET-MAD-XS zero-shot on MD17/sGDML's CCSD(T) ethanol dataset, run
  molecular dynamics with it, then fine-tune it live on a few hundred structures and repeat
  everything. The notebook builds towards a deliberately hard question: the few-meV energy
  difference between the *anti* and *gauche* conformers, which the zero-shot model gets
  qualitatively wrong and fine-tuning largely fixes. About 13 minutes of compute end to end
  on a GPU, ~45 minutes including discussion.
* [`water-md/water_md.ipynb`](water-md/water_md.ipynb) — **zero-shot condensed phase**.
  Molecular dynamics of liquid water and an NaCl solution in LAMMPS (radial distribution
  functions, coordination numbers), then superionic water at 3000 K and ~130 GPa, where the
  mean-squared displacement shows oxygen frozen on its lattice while hydrogen diffuses like a
  liquid — four orders of magnitude apart in the same box. Short runs happen live, the
  analysis uses longer precomputed trajectories; ~45 minutes. Adapted from Paolo Pegolo's
  [atomistic-cookbook recipe](https://atomistic-cookbook.org).

Both notebooks share the same model checkpoint, `pet-mad-xs-v1.6.0.ckpt`. It is **not** in
this repository — it lives on the classroom machine, at
`/home/unito/pet-mad-models/`. The notebooks make opposite points and complement each
other: the ethanol notebook is about specialising a foundation model when you have
reference data, the water notebook about using one unchanged when you have none.

An earlier aspirin version of the fine-tuning notebook is kept in
`deprecated/aspirin-finetune/` for reference. It is not part of the session and is not
maintained.

## Setup

```bash
git clone <this-repo-url>
cd <this-repo>
conda env create -f environment.yml
conda activate pet-mad-hands-on
jupyter lab
```

That's it — no other download or build step on the classroom machine, where the checkpoint
is already in place. Running anywhere else, put your own copy somewhere and change the
`CKPT_PATH` line near the top of each notebook to point at it. `environment.yml` pins
everything needed:
`metatrain` (to load/export the checkpoint and to run the live fine-tuning step),
`metatomic-ase` + `ase` (to run the ethanol MD), and a prebuilt `lammps-metatomic` conda
package (to run the water/NaCl MD) — no LAMMPS compilation required.

Both notebooks pick a device automatically (`DEVICE = "cuda" if torch.cuda.is_available()
else "cpu"`) and use it everywhere — the LAMMPS runs and the fine-tuning step.
`environment.yml` installs a CUDA-enabled `lammps-metatomic` build (Ampere / compute
capability 8.6, matching the NVIDIA A16) that also runs fine with no GPU at all, so the
same environment works on a laptop or on the shared classroom machine.
