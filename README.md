# ConnectX Kaggle

A high-performance Connect Four (ConnectX) game solver featuring multiple AI strategies optimized for different game phases. This project combines opening books, AlphaZero, and advanced minimax algorithms to create a competitive AI player.

## Overview

The AI uses a hybrid approach, selecting different strategies based on the game state:
- **Opening Book** (first 8 moves): Pre-computed optimal moves for early game
- **AlphaZero** (moves 8-17): AlphaZero for mid-game play
- **Alpha-Beta Minimax** (moves 17+): Advanced minimax with bitboard optimization for endgame

## Project Structure

```
connectx/
├── src/
│   ├── main.py                                    # Main entry point (hybrid strategy selector)
│   ├── best.py                                    # Alpha-Beta Minimax with Bitboards (endgame)
│   ├── submission.py                              # AlphaZero (mid-game)
│   └── opening_book.py                            # Opening book implementation
├── benchmark/
│   ├── benchmark_moves.py                         # Move generation benchmarks
│   ├── benchmark_score.py                         # Scoring benchmarks
│   ├── book_generate.py                           # Opening book generation tool
│   ├── benchmark_results/                         # Performance comparison results
│   ├── data_set/                                  # Test positions
│   └── c4solver                                   # External solver for reference
├── test/
│   ├── test.py                                    # Simulation
│   ├── test_single.py                             # Single game tests
│   └── test_multiple.py                           # Multiple game tests
└── README.md
```

## Algorithm Implementations

### 1. Alpha-Beta Minimax
The primary endgame solver using:
- **Alpha-Beta Pruning**: Eliminates branches that cannot affect final decision
- **Move Exploration Order**: Prioritize middle column outward.
- **Bitboard Encoding**: Compact board representation (64-bit) for O(1) operations
- **Transposition Tables**: Memoization of evaluated positions for reuse
- **Anticipate Losing Moves**: Avoid moves that immediately allow forced wins for the opponent.

### 2. AlphaZero (`submission.py`)
Mid-game strategy using AlphaZero and Monte Carlo Tree Seacrh

### 3. Opening Book (`opening_book.py`)
Early game optimization:
- Pre-computed optimal moves for opening positions
- Reduces branching factor in first 8 moves
- Ensures consistent, strong opening play

## Getting Started

### Prerequisites
- Python
- NumPy (for board representation and array operations)
- Kaggle Environment (for testing)

### Installation

```bash
# Clone the repository
git clone https://github.com/LeonaRaging/connectx
cd connectx

# Install dependencies
pip install numpy
pip install kaggle-environments
```

## Usage

### Playing Against the AI
```bash
python -m test.test
```
Enter empty line if you want the AI to go first else enter column index.

### Running Tests

```bash
# Run all tests
python test/test.py

# Test single game
python test/test_single.py

# Test multiple games
python test/test_multiple.py
```

### Benchmarking

```bash
# Benchmark different algorithms
python benchmark/benchmark_score.py
python benchmark/benchmark_moves.py

# Generate opening book
python benchmark/book_generate.py

# View benchmark results
cat benchmark/benchmark_results/
```

### Resources
[Geeksforgeks: Monte Carlo Tree Search in Machine Learning](https://www.geeksforgeeks.org/machine-learning/monte-carlo-tree-search-mcts-in-machine-learning/)

[Connect 4: How to build a perfect AI](http://blog.gamesolver.org/solving-connect-four/01-introduction/)


