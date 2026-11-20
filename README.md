# Student pass/fail predictor - Shiny for Python (Machine Problem 3)

Predicts whether a student is likely to pass or fail from LMS activity.
Model: Random Forest (300 trees) trained on the Machine Problem 2 dataset.
The dataset's `High` rating is read as **Pass** and `Low` as **Fail**.

## Files
| File | Purpose |
|---|---|
| `app.py` | The Shiny app (the primary file when deploying) |
| `train_model.py` | Removes duplicates, trains and evaluates the model, saves `model.joblib` and `metrics.json` |
| `augmented_balanced_dataset_m.csv` | Dataset from Machine Problem 2 |
| `model.joblib`, `metrics.json` | Saved model and its evaluation numbers |
| `requirements.txt` | Python packages (Connect Cloud needs this file) |

## Run locally
```
pip install -r requirements.txt
shiny run app.py --reload
```
Then open http://localhost:8000

## Deploy on Posit Connect Cloud (free)
1. Create a **public** GitHub repository and upload every file in this folder to the repo root.
2. Go to https://connect.posit.cloud and sign up / sign in with GitHub.
3. Click **Publish**, choose **Shiny**, and allow the GitHub App if asked.
4. Select your repository and branch (usually `main`), and set the **primary file** to `app.py`.
5. Under **Project files**, check the list includes `app.py`, `train_model.py`, `model.joblib`,
   `metrics.json`, `augmented_balanced_dataset_m.csv` and `requirements.txt`.
   Connect Cloud only uploads the files in that list.
6. Click **Publish**. When the build finishes you get a URL to paste into the submission.

## Results (duplicates removed before splitting)
- 2,720 rows after removing 882 exact duplicates
- 5-fold cross-validated accuracy: about 80.7%
- Hold-out test accuracy (80/20 stratified split): about 76.7%
