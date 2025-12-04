# heart_failure_project

# ❤️ Heart Failure Prediction using Support Vector Machine (SVM)

This project applies **Machine Learning (ML)** techniques to predict **Heart Disease** using the Support Vector Machine (SVM) classifier.  
The goal is to analyze clinical features and classify patients into **heart disease** or **healthy**, supporting early detection and medical decision-making.

---

## 📌 Project Overview
Heart failure is one of the leading causes of death worldwide. Early detection improves the chances of effective treatment.  
This project uses **clinical data** and machine learning to build a predictive model that identifies individuals who may have heart disease.

The model is implemented in Python using:
- **pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**

---

## 🧠 Machine Learning Model: SVM

Support Vector Machine (SVM) is a robust classification algorithm.  
It works by:
- Finding the optimal **hyperplane** that separates the two classes.
- Maximizing the **margin** between classes.
- Supporting linear and non-linear decision boundaries.

We used:
```python
from sklearn.svm import SVC
SVC(max_iter=5000)


📊 Dataset Description

The dataset includes clinical attributes such as:

Feature	Description
Age	Age of the patient
Sex	M = Male, F = Female
ChestPainType	TA, ATA, NAP, ASY
RestingBP	Resting blood pressure
Cholesterol	Serum cholesterol
FastingBS	Fasting blood sugar
RestingECG	ECG results
MaxHR	Maximum heart rate
ExerciseAngina	Exercise-induced angina
Oldpeak	ST depression
ST_Slope	Up, Flat, Down
HeartDisease	1 = disease, 0 = normal
🛠️ Technologies Used

Python

Google Colab / Jupyter Notebook

Scikit-learn

Pandas

Seaborn

Matplotlib

🧹 Data Preprocessing

✔ Handling missing values
✔ Detecting duplicates
✔ Encoding categorical features
✔ Descriptive statistics
✔ Data visualization

Example encoding:

Heart['Sex'] = Heart['Sex'].map({'M':1, 'F':0})
Heart['ExerciseAngina'] = Heart['ExerciseAngina'].map({'Y':1, 'N':0})
Heart['ChestPainType'] = Heart['ChestPainType'].map({'TA':3, 'ATA':2, 'NAP':1, 'ASY':0})
Heart['ST_Slope'] = Heart['ST_Slope'].map({'Up':2, 'Flat':1, 'Down':0})
Heart['RestingECG'] = Heart['RestingECG'].map({'Normal':1, 'ST':2, 'LVH':3})

🚀 Model Training
from sklearn.svm import SVC

sm = SVC(max_iter=5000)
sm.fit(X_train, y_train)
y_pred = sm.predict(X_test)

📈 Model Evaluation
✔ Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

✔ Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

📝 Conclusion

The SVM model achieved strong performance in predicting heart disease based on clinical data.
Its high accuracy and reliable classification metrics make it a powerful tool for early detection and biomedical applications.
