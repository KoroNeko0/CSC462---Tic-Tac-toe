# Tic-Tac-Toe Move Prediction

A machine learning project that predicts the optimal next move in Tic-Tac-Toe from a board state, framed as a 9-class classification problem. Done for CSC462 - Machine Learning at King Saud University.

## Overview

Tic-Tac-Toe is a solved game, but this project simulates a real ML task: models are trained only on board states and Minimax-labeled "best move" examples, without being told the rules, and must generalize to a hidden test set.

**Input:** 9 board cells (p1–p9, each -1/0/1) plus a turn indicator.
**Output:** predicted best move (cell index 0–8).

Four models were built, progressing from a simple linear baseline to more advanced ensemble and neural approaches, with a focus on minimizing the train-validation gap rather than maximizing training accuracy.

## Dataset

- 3,164 training examples, 678 validation examples, plus a hidden test set.
- Clean data: no missing values or duplicates, all features within expected ranges.
- Labels are imbalanced — cell 0 (top-left corner) is most common (765 examples), cell 7 (bottom-middle edge) is rarest (117 examples).

## Models and Results

| Model | Feature Engineering | Train Acc | Val Acc | Train-Val Gap |
|---|---|---|---|---|
| Logistic Regression | None | 25.06% | 21.98% | 3.08% |
| Logistic Regression | Polynomial (degree 2) | 85.78% | 81.42% | 4.36% |
| Random Forest | Strategic features | 82.71% | 76.40% | 6.31% |
| MLP | Strategic features | 93.14% | 85.84% | 7.30% |
| XGBoost | Strategic features + L2 | 90.30% | 86.28% | 4.01% |

**XGBoost** was the best overall model: highest validation accuracy (86.28%) with the smallest train-validation gap (4.01%), indicating strong generalization rather than memorization.

## Key Findings

- The plain Logistic Regression baseline severely underfits, since it evaluates each cell independently and can't capture multi-cell tactical patterns.
- Adding polynomial (interaction) features let Logistic Regression jump from ~22% to ~81% validation accuracy — how the data is represented mattered as much as the model choice.
- Game-aware engineered features (two-in-a-row counts, fork detection, center/corner control, empty-cell count) consistently improved both accuracy and generalization across all advanced models.
- XGBoost and MLP were the strongest models, both exceeding 85% validation accuracy.

## Feature Engineering

Strategic features added on top of the 10 raw inputs include:
- **Two-in-a-row counts** for both the player and opponent (winning/blocking opportunities)
- **Fork detection** — whether a player has two or more simultaneous threats
- **Center and corner control** indicators
- **Empty cell count** as a game-phase signal

## Requirements

- Python 3
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- xgboost
- joblib

## Repository Structure

```
.
├── data/               # train.csv, val.csv
├── notebooks/          # model training and evaluation code
├── models/             # saved model files (.pkl)
├── report/             # full project report (PDF)
└── README.md
```

## Running

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

Then run the notebook/script to load the data, train each model, and reproduce the accuracy and log-loss metrics above.

## Future Work

- Expand the engineered feature set further.
- Increase MLP capacity and tune architecture.
- Use grid search / systematic hyperparameter tuning for each model.

Anfal Alobeid, Rama Alfawzan, Maha Aldakhil, Batool Alyousef — King Saud University, CSC462.
