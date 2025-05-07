# Music Genre Classification

112-2 Multimedia Content Analysis Project2

🚀 check out the [report](https://github.com/JenLungHsu/Music-Genre-Classification/blob/main/Music%20Genre%20Classification.pdf)  for more detail.

## Project Overview
This project focuses on the classification of music genres using various **feature extraction methods** and **classification models**. The primary objective is to evaluate how different combinations of feature extraction techniques and models impact the accuracy of genre classification. The implementation covers feature extraction, model training, cross-validation, and final evaluation on the GTZAN dataset, which consists of 10 distinct music genres.

## Dataset
- **Dataset:** GTZAN Music Genre Dataset
- **Description:** The dataset contains 10 genres, each with 50 audio clips, each lasting 30 seconds.
- **Download:** [Google Drive Link](https://drive.google.com/drive/folders/17Byto32o58zpmHyJBKhZz_pjZcdbGZTr?usp=sharing)

---

## Project Structure
```
├── Music Genre Classification.ipynb   # Jupyter Notebook for full training and evaluation
├── Music Genre Classification.pdf     # Report of the analysis and findings
├── requirements.txt                   # Dependencies for running the notebook
└── README.md                          # Project documentation
```

---

## Methodology
1. **Feature Extraction:**
   - Mel-Frequency Cepstral Coefficients (MFCC)
   - Fast Fourier Transform (FFT)
   - Rhythm Features (tempo, beat, rhythm patterns)
   - Pitch Features (pitch chroma, pitch histogram)
   - Combination of all features

2. **Classification Models:**
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - Adaptive Boosting (AdaBoost)
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors (KNN)

3. **Strategy:**
   - Among various combinations of feature extraction methods and models, the best performance was found when **all features** were combined with **Random Forest**. This configuration was further fine-tuned through hyperparameter optimization, leading to an improved final test accuracy. Parameters such as `n_estimators`, `max_depth`, and `min_samples_split` were adjusted to enhance model performance.

4. **Cross-Validation:**
   - 5-fold cross-validation was applied to evaluate model stability and avoid overfitting.

---

## Results
The best performance was achieved with:
- **Model:** Random Forest (with optimized hyperparameters)
- **Feature Set:** Combination of all features (MFCC, FFT, Rhythm, Pitch)
- **Test Accuracy:** 0.69

---

## Usage
To execute the notebook:
```bash
pip install -r requirements.txt
jupyter notebook 'Music Genre Classification.ipynb'
```

Before running the notebook, download the dataset from the [Google Drive link](https://drive.google.com/drive/folders/17Byto32o58zpmHyJBKhZz_pjZcdbGZTr?usp=sharing) and place it in the appropriate folder.

---

## Contact
- **Author:** Jen Lung Hsu
- **Email:** RE6121011@gs.ncku.edu.tw
- **Institute:** National Cheng Kung University, Institute of Data Science
