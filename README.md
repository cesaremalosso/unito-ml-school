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

Both notebooks share the same model checkpoint, `pet-mad-xs-v1.6.0rc4.ckpt`, shipped at the
repository root. They make opposite points and complement each other: the ethanol notebook
is about specialising a foundation model when you have reference data, the water notebook
about using one unchanged when you have none.

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

That's it — no other download or build step. `environment.yml` pins everything needed:
`metatrain` (to load/export the checkpoint and to run the live fine-tuning step),
`metatomic-ase` + `ase` (to run the ethanol MD), and a prebuilt `lammps-metatomic` conda
package (to run the water/NaCl MD) — no LAMMPS compilation required.

Both notebooks pick a device automatically (`DEVICE = "cuda" if torch.cuda.is_available()
else "cpu"`) and use it everywhere — the LAMMPS runs and the fine-tuning step.
`environment.yml` installs a CUDA-enabled `lammps-metatomic` build (Ampere / compute
capability 8.6, matching the NVIDIA A16) that also runs fine with no GPU at all, so the
same environment works on a laptop or on the shared classroom machine.

## Notes for instructors

### General

* **Timing knobs.** Both notebooks expose the expensive step counts as variables near the
  top of each section (`nsteps` in the `.lmp` files, `MD_STEPS` and `NUM_EPOCHS` in the
  ethanol notebook) — turn them down if your classroom hardware is slower than expected, or
  up if you have time to spare. Measure them once on the machine you will actually teach
  on: the numbers below were measured on an RTX A6000 and CPU-only runs are much slower.
* Both notebooks derive the model file they actually use (`pet-mad-xs.pt`) from the shipped
  `.ckpt` via `metatrain`'s export in an early cell — this also strips the checkpoint's
  built-in uncertainty-quantification (LLPR) wrapper, which is not needed here and roughly
  doubles inference cost.

### The ethanol notebook

* **Structure.** Nine sections: look at the data, split it, load the model zero-shot, test
  it on held-out CCSD(T) energies and forces, relax + run MD, compare the MD structure with
  the reference, fine-tune, re-measure everything, and finally the conformer-energetics
  section. Cells marked **Your turn** (11 of them) hold the exercises — choosing the
  train/validation/test split, choosing the MD timestep, choosing the number of epochs, and
  a number of predict-before-you-run prompts.
* **What the numbers come out at**, with the shipped defaults (500 training structures, 15
  epochs, ~2 minutes of fine-tuning on a GPU):

  | | zero-shot | fine-tuned |
  |---|---|---|
  | energy MAE | 2.68 meV/atom | 0.71 meV/atom |
  | force MAE | 87.6 meV/A | 17.5 meV/A |
  | E(gauche) - E(anti) | +31.0 meV | -4.9 meV |

  Expect small run-to-run variation; the qualitative picture is stable.
* **The conformer section is the point of the notebook.** It makes the same measurement
  three independent ways, and they agree:
  1. a relaxed scan of the C1-C2-O-H torsion with each model;
  2. the model-minus-CCSD(T) residual as a function of that same dihedral, computed on
     thermal dataset structures — completely different geometries, same answer (a bias of
     +28 meV zero-shot, -4 meV fine-tuned);
  3. a mirror-image test: gauche+ and gauche- are mirror images and must have identical
     energies, so the measured difference (RMS 18.5 meV zero-shot, 5.3 meV fine-tuned) is
     pure model error and sets the resolution limit of the whole exercise.

  The honest conclusion the notebook draws is that neither model can resolve the *sign* of
  the anti/gauche difference, and that this is a good result rather than a disappointing
  one: the value of sections 9.3 and 9.4 is knowing what the number in 9.2 is worth. Worth
  reinforcing out loud — students tend to want the pretty curve to be the answer.
* **The MD is NVE on purpose**, with velocities seeded at twice the target temperature
  (starting from the minimum, equipartition hands about half the kinetic energy to the
  potential, so seeding at 2T lands near T) and net translation and rotation removed. No
  thermostat means the total energy is a conserved quantity students can watch — and
  deliberately break, in the timestep exercise, where 0.5 fs, 1 fs, 2 fs and 4 fs span the
  whole range from a bounded few-meV wobble to a run that blows up by tens of thousands of
  eV. The framing that lands is *steps per C-H period* (~11 fs): 22, 11, then fewer than 6.
  Note that near the stability threshold the outcome depends on the initial velocities, so
  a 2 fs run may survive one seed and explode on the next.
* **The MD runs at 500 K, matching the temperature of the reference dataset.** This matters
  more than it sounds: bond lengths and angles are anharmonic, so their averages shift with
  temperature. Run the same comparison at 300 K and the mean C1-C2-O angle looks 4 degrees
  "wrong" when nothing is wrong with the potential. It is a cheap and memorable lesson in
  comparing like with like.
* **Expect fine-tuning to barely improve the bond lengths.** It improves forces by a factor
  of five and leaves the stiff coordinates essentially where they were; the structural
  table carries blocked error bars so students can see which changes are significant. This
  is the setup for section 9 — a model gets better on the *soft* observables, not uniformly.
* An earlier version of this notebook compared a computed vibrational spectrum against a
  digitised NIST gas-phase IR measurement. That comparison has been removed: the shipped
  CSV turned out to be unusable (non-monotonic wavenumber axis, no intensity in the C-H
  stretch region around 2850-3000 cm^-1, a spurious band near 4143 cm^-1 above any ethanol
  fundamental). If you want the comparison back, start by re-downloading the JCAMP-DX file
  from the [NIST WebBook](https://webbook.nist.gov/cgi/cbook.cgi?ID=C64175&Type=IR-SPEC)
  and checking it against known band positions before plotting anything against it.

### The water notebook

* **Live runs versus precomputed ones.** All three live systems (water, NaCl, the
  ethanol-water appendix) run 2,000 steps = 1 ps, measured at ~41 ms/step on an RTX A6000,
  so about 80 seconds each. That is enough to watch the dynamics and confirm stability, and
  nowhere near enough for converged statistics — so the *analysis* uses the longer
  trajectories in `water-md/trajs/` (10 ps for water and NaCl, 20 ps for the two superionic
  runs), produced with the same inputs. Section 2 of the notebook compares the 1 ps and 10 ps
  oxygen-oxygen g(r) side by side to make the point that this is a decision, not an
  accident.
* `nsteps` is declared **index-style** in every `.lmp` file, so the notebook overrides it with
  `-var nsteps N` (`LIVE_STEPS` in the setup cell). Equal-style variables cannot be
  overridden from the command line — LAMMPS errors out — which is why they were changed.
* **Resource note.** PET-MAD's fairly large interaction cutoff (7.5 A) means the periodic
  water/NaCl boxes (~190 atoms in a 12.5 A cell) cost far more per step than a small
  non-periodic molecule: every atom sees every other atom, and then some through the periodic
  images.
* **The superionic trajectories are precomputed on purpose** (384 atoms, high pressure,
  20 ps each — long enough for hydrogen to visibly diffuse across many lattice sites) and
  are shipped alongside the notebook in `water-md/trajs/` — do not try to reproduce them
  live with 50 people at once. They took ~3.5 hours each on an RTX A6000; the input files
  (`in_superionic_*.lmp`) are shipped so students can read exactly how they were produced.
  They are dumped every 50 steps (10 fs), giving ~2000 frames over 20 ps — fine enough to
  resolve the short-time ballistic regime of the MSD, at ~25 MB per trajectory.
