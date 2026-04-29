# Technical Paper Presentation Predictor

A simple Machine Learning project that predicts whether a student's technical paper presentation performance is **Low**, **Medium**, or **High** based on 8 evaluation parameters.

---

## Project Files

| File | Purpose |
|------|---------|
| `generate_dataset.py` | Creates the synthetic CSV dataset |
| `train_model.py` | Trains the ML model and saves it |
| `app.py` | Streamlit web UI for prediction |
| `requirements.txt` | Python dependencies |

---

## How to Run

### Step 1 — Install dependencies
Open a terminal in this folder and run:
```
pip install -r requirements.txt
```

### Step 2 — Generate the dataset
```
python generate_dataset.py
```
This creates `dataset.csv` in the same folder.

### Step 3 — Train the model
```
python train_model.py
```
This trains the Random Forest model and saves it as `model.pkl`.  
You will also see the accuracy and confusion matrix printed in the terminal.

### Step 4 — Launch the UI
```
streamlit run app.py
```
A browser window will open automatically with the prediction interface.

---

## Input Features

| Feature | Description | Range |
|---------|-------------|-------|
| Paper Quality | Quality of the written paper | 1–10 |
| Presentation Skills | How well the student presented | 1–10 |
| PPT Design | Visual quality of the slides | 1–10 |
| Content Clarity | How clearly content was explained | 1–10 |
| Technical Depth | Depth of technical knowledge shown | 1–10 |
| Q&A Handling | How well questions were answered | 1–10 |
| Time Management | Whether time limits were respected | 1–10 |
| Confidence Level | Confidence shown during presentation | 1–10 |

## Output

| Category | Meaning |
|----------|---------|
| Low | Score below 40 — Poor performance |
| Medium | Score 40–70 — Average performance |
| High | Score above 70 — Excellent performance |

---

## Tech Stack
- Python 3.x
- pandas, numpy
- scikit-learn (Random Forest)
- Streamlit
