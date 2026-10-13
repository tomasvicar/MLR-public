# Leaves k-NN — starter project for exercise 5

A small, working Python project: the k-nearest-neighbours classifier from
exercises 1 and 2, split into a module, a script and a test file. It is the
code base you will put under git and then extend together with a coding agent
in [exercise 5](https://tomasvicar.github.io/MLR-public/exercises/ex05.html).

Everything here runs and every test passes. That is the point: you start from
a green test suite, so you can tell what the agent broke.

## Files

| File | What is in it |
| --- | --- |
| `knn.py` | `euclidean_distances`, `standardize`, and the `KNNClassifier` class with `fit` / `predict` |
| `evaluate.py` | loading the leaves, `train_test_split`, `accuracy`, and a `main()` that prints the test accuracy |
| `test_knn.py` | six pytest tests on small hand-made arrays — no internet needed |
| `leaves.csv` | the data (518 leaves) |
| `requirements.txt` | numpy, pandas, pytest |

## Running it

With [uv](https://docs.astral.sh/uv/) (nothing to install first):

```bash
uv run --with numpy --with pandas python evaluate.py
uv run --with numpy --with pandas --with pytest pytest
```

Or in a plain virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python evaluate.py
python -m pytest
```

Expected output of `python evaluate.py`:

```
363 training and 155 test leaves
k = 5   test accuracy = 0.9548
```

## Data

`leaves.csv` — a copy of
[`exercises/data/leaves.csv`](https://raw.githubusercontent.com/tomasvicar/MLR-public/master/exercises/data/leaves.csv),
518 leaves measured by the students of this course, columns `Leaf height`
[cm], `Leaf width` [cm] and `Tree type` (216 maple, 173 cherry, 129 walnut).
The same table as in exercises 1, 2 and 6.
