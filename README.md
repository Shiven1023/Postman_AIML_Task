# Sparse Attention from Scratch

This project implements and compares:

* Dense causal attention
* Sliding-window attention
* Block-sparse attention
* BigBird-style sparse attention

It also contains correctness tests, runtime/memory benchmarks, and a small 2-layer character GPT trained on TinyShakespeare.

## Run Correctness Tests

From the repository root:

```bash id="ztkcux"
python3 -m tests.correctness
```

## Run Benchmarks

Run the benchmark scripts from the repository root, for example:

```bash id="u4fglf"
python3 -m benchmarks.benchmark
```

## Run TinyShakespeare Training

The dataset should be located at:

```text id="ffogme"
data/tiny-shakespeare.txt
```

Run:

```bash id="llcsli"
python3 -m train.model
```

The script trains the same 2-layer character GPT using:

* Dense attention
* Sliding-window attention
* BigBird-style attention

and prints the final validation loss for each.

Training configuration details such as batch size, context length, embedding dimension, number of heads, learning rate, and training steps are defined directly in `train/model.py`.

CUDA is used automatically if available.

## Requirements

```bash id="k2hmdd"
pip install torch matplotlib memory-profiler
```
