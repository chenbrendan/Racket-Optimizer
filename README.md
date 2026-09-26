# RacketFEA: Stringing Sequence Optimizer

RacketFEA compares the frame response during different racket stringing orders. It models a racket frame with planar finite elements, evaluates the frame after each string is installed, and searches for a feasible order with a lower weighted deformation and bending score.

This is a simplified comparison model. Its results are not a validated recommendation for stringing a real racket.

## Model and search

- The default frame is an ellipse made of 120 two-dimensional Euler-Bernoulli beam elements. Each node has two translations and one rotation.
- The default string pattern has 16 mains and 18 crosses, each prescribed at 100 N.
- The default mount has six support points represented by radial springs, plus constraints to remove rigid-body motion. A throat-clamp option is also available.
- The fast search solves the frame response to each string separately. Linear superposition then gives the intermediate frame state after each successive pull.
- The baseline strings the mains center-out, followed by the crosses center-out. Feasible candidates finish all mains before crosses, progress outward on each side, and use no more than two consecutive strings on the same side.
- Simulated annealing searches by swapping adjacent strings within those constraints. The default run uses 3,000 iterations per restart and four restarts. The search finds an improved candidate when available; it does not prove global optimality.

The objective is a weighted sum of six metrics measured across the intermediate stringing states: peak nodal displacement, peak beam-end bending moment, peak ovalization, peak asymmetry, peak transient shape change, and average peak displacement across stages (deformation exposure). Each metric is normalized against the fully strung frame response calculated by the model. The weights encode a modeling choice, not measured damage thresholds.

With prescribed string forces and linear frame stiffness, the final frame state is independent of stringing order. The search compares **intermediate states**.

## Run it

Use Python with NumPy and Matplotlib:

```bash
python -m pip install numpy matplotlib
python racket_fea.py
```

The script prints the baseline and best-found scores, selected deformation metrics, percentage improvement, and the best sequence as **zero-based string indices**. Indices 0–15 are mains and 16–33 are crosses with the default pattern, in the order the strings are created from negative to positive coordinates.

It also saves `fea_result.png`. The left plot shows the best score found during the search against the baseline; the right plot shows the **undeformed** frame, string layout, and support locations. The plot is saved even if `--show` is specified.

Useful options:

```bash
python racket_fea.py --help
python racket_fea.py --elastic-check
python racket_fea.py --nodes 120 --mains 16 --crosses 18 --tension 100 --iterations 3000 --restarts 4 --seed 7 --support six_point --plot fea_result.png
```

Use `--support throat_clamp` to switch mount models. The optional `--elastic-check` evaluates the **best-found sequence after the search**, treating installed strings as axial springs and printing the final modeled tension range. It does not change the optimizer's objective or search.

Run the included tests with:

```bash
python -m unittest test_racket_fea.py
```

## Interpretation and limits

The ellipse, uniform beam properties, idealized string attachments, mount, and objective weights are assumptions. A real composite racket has varying geometry and material behavior, as well as effects such as contact, grommet friction, and tension loss that this model does not calibrate. Compare candidate orders **within these assumptions**; do not interpret a small score difference as proof of a physical performance or durability benefit.
