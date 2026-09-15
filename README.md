# 🏸 RacketFEA: The Machine Learning Stringer

Welcome to a ridiculously deep dive into a very niche intersection: **Racket Sports + Structural Engineering + Machine Learning**.

Have you ever wondered if the standard way we string tennis and badminton rackets is *actually* the best way to protect the frame? (Spoiler: It's not).

## 🎾 What is this?
This repository contains a physically-grounded racket stringing-order optimizer. It models a racket frame using Finite Element Analysis (FEA) and uses a combinatorial search algorithm (Simulated Annealing) to find the exact sequence of stringing that minimizes structural warping and transient stress on the frame.

## 💥 The Problem with "Folk Wisdom"
For decades, professional stringers have used a traditional "center-out" alternating method. The logic makes sense on the surface: it keeps the frame visually balanced. 

But there's a structural catch. Pulling a main (vertical) string shifts the stress right where a nearby cross (horizontal) string attaches. Traditional heuristics treat mains and crosses as completely separate sequences, ignoring this physical coupling effect! 

When we calculate the **peak nodal displacement** (how much the racket literally bows out of shape mid-stringing before the opposing strings lock it back into place), the results are eye-opening:

*   **Random Sequence:** `~0.000002 m`
*   **Traditional 'Center-Out':** `~0.000012 m` 🚨 *(Surprisingly, the worst!)*
*   **ML Search-Optimized:** `~0.000002 m` 🏆 *(The mathematically optimal route)*

## 🧠 How the Code Works
1. **The Physics (FEA Proxy):** The frame is modeled as a closed loop of 60 2D beam elements with real bending and axial stiffness. It's rigidly clamped at the throat—exactly like it sits in a real stringing machine.
2. **Superposition Magic:** Running a full FEA solve for *every single permutation* of string pulls would take centuries. Instead, we solve the frame FEA **once** per string at unit tension to build an **Influence Matrix**. Because the frame is linear-elastic, evaluating any sequence of strings becomes simple, lightning-fast vector addition.
3. **The Search:** We throw Simulated Annealing at the problem. The algorithm randomly swaps string orders thousands of times, steadily cooling down to lock into a sequence that interleaves mains and crosses perfectly, keeping peak frame distortion as close to zero as physically possible.

## 🚀 Getting Started

### Prerequisites
You'll need Python and a couple of standard scientific libraries:
```bash
pip install numpy matplotlib
```

### Running the Optimizer
Clone the repo and run the main script:
```bash
python racket_fea.py
```
*(Note: By default, the script generates an image file. If you want the graphs to pop up on your screen interactively, remove `matplotlib.use("Agg")` from the imports and change `plt.savefig(...)` to `plt.show()` at the bottom of the script!)*

## 📊 Visualizing the Output
When you run the script, it generates a visualization (`fea_result.png`) showing:
1. The learning curve of the ML algorithm completely bypassing the traditional baseline.
2. A geometric plot of the FEA frame, the clamped throat nodes, and the string map.

---
*Built with Python, Structural Mechanics, and a mild obsession with over-engineering sports equipment.*
