#  Tennis Match Winner Prediction using Deep Learning

A deep learning project that predicts the eventual winner of a tennis match from the sequence of points played using a Long Short-Term Memory (LSTM) network implemented in **PyTorch**.

The model observes the progression of a match point-by-point and continuously estimates each player's probability of winning the match.

---

## Project Overview

Predicting the winner of a tennis match is a challenging sequential learning problem. Instead of making predictions from only the current score, this project models the **entire history of the match** up to the current point.

Each point played contributes to a sequence describing the current state of the match. These sequences are used to train an LSTM capable of learning momentum, score progression, and temporal dependencies throughout a match.

---

##  Features

- Point-by-point match preprocessing
- Feature engineering from raw tennis scores
- Variable-length sequence generation
- Efficient sequence packing using `pack_padded_sequence`
- LSTM implemented using PyTorch
- Live match winner prediction
- Win probability estimation for both players
- Model evaluation using multiple performance metrics

---

# Dataset

The project uses the **Tennis Match Charting Project** dataset containing point-by-point information from professional ATP matches.

The dataset includes:

- Current set score
- Current game score
- Point score
- Server
- Point winner
- Match winner

Only publicly available match statistics are used.

---

# Feature Engineering

The following features are used as inputs to the LSTM:

| Feature | Description |
|----------|-------------|
| Set1 | Sets won by Player 1 |
| Set2 | Sets won by Player 2 |
| Gm1 | Games won by Player 1 |
| Gm2 | Games won by Player 2 |
| P1_pts | Current point score of Player 1 |
| P2_pts | Current point score of Player 2 |
| Svr | Current server |
| set_diff | Set difference |
| game_diff | Game difference |
| point_diff | Point difference |


---

# Model Architecture

```
Input Sequence
        │
        ▼
LSTM Layer
        │
        ▼
Final Hidden State
        │
        ▼
Fully Connected (32 → 16)
        │
        ▼
ReLU
        │
        ▼
Dropout
        │
        ▼
Fully Connected (16 → 2)
        │
        ▼
Softmax
        │
        ▼
Win Probability
```

---

# Training Pipeline

1. Data cleaning
2. Feature engineering
3. Train/Test split by match
4. Variable-length sequence generation
5. Dynamic padding using a custom DataLoader
6. LSTM training
7. Validation
8. Save best model

---

# Repository Structure

```
tennis-match-winner-prediction/
│
├── notebooks/
│   ├── 01_Data_Preprocessing.ipynb
│   ├── 02_Model_Training.ipynb
│   ├── 03_Model_Evaluation.ipynb
│   └── 04_Live_Match_Prediction.ipynb
│
├── models/
│   └── best_tennis_lstm.pth
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/arjun1151/live-tennis-match-winner-prediction-model.git
```

Move into the project directory

```bash
cd live-tennis-match-winner-prediction-model
```


# Training

Run the notebooks in order:

```
01_Data_Preprocessing.ipynb

↓

02_Model_Training.ipynb

↓

03_Model_Evaluation.ipynb

↓

04_Live_Match_Prediction.ipynb
```

---

# Model Performance

| Metric | Score |
|---------|------:|
| Best Validation Accuracy | **71.33%** |
| Precision | *0.7* |
| Recall | *0.7* |
| F1 Score | *0.69* |

---

# Live Match Prediction

The trained model can estimate the probability of each player winning while the match is in progress.

Example:

```
Enter each point as:
Set1 Set2 Gm1 Gm2 P1_pts P2_pts Svr

Please enter exactly 7 values.
[1, 0, 4, 3, 0, 0, 2, 1, 1, 0] 

[1, 0, 4, 3, 1, 0, 2, 1, 1, 1] 

[1, 0, 4, 3, 1, 1, 2, 1, 1, 0] 

[1, 0, 4, 3, 2, 1, 2, 1, 1, 1] 

[1, 0, 4, 3, 2, 3, 2, 1, 1, -1] 


```

Output:

```

Player 1 win probability: 67.56%
Player 2 win probability: 32.44%
```

---

# Results

The model successfully learns temporal patterns from point-by-point match data and achieves approximately **71% validation accuracy** on unseen matches.

Unlike traditional classifiers that only use the current score, the LSTM captures historical match progression, making it suitable for live win prediction.

---

# Future Improvements

- Transformer-based sequence models
- Attention mechanisms
- Surface-specific models (Clay, Grass, Hard)
- ATP ranking and Elo rating features
- Player-specific embeddings
- Transfer Learning 

---

# Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook
- Seaborn

---

# Acknowledgements

- Tennis Match Charting Project
- PyTorch Documentation
- Scikit-learn Documentation

---
