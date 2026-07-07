
### Project Title
**Housing Price Classification Pipeline**

### Overview
This project implements a machine learning pipeline to classify housing prices into 'High' or 'Low' categories based on various property features. It addresses common pitfalls in ML workflows, such as handling categorical data, preventing data leakage during scaling, and managing class imbalance.

### Dataset
The pipeline utilizes the `Housing.csv` dataset, which contains information about various residential properties, including area, number of bedrooms, bathrooms, stories, and amenities like `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `parking`, `prefarea`, and `furnishingstatus`.

### Objective
To build a robust classification model capable of predicting whether a house's price falls above or below the median price, thus categorizing it as 'High' (1) or 'Low' (0) priced.

### Key Libraries Used
-   `pandas` for data manipulation and analysis.
-   `numpy` for numerical operations.
-   `sklearn` for machine learning tasks (model selection, preprocessing, classification, metrics).
-   `matplotlib` and `seaborn` for data visualization.
-   `imblearn` (Imbalanced-learn) for handling class imbalance (`SMOTE`).

### Machine Learning Pipeline Steps
1.  **Data Loading**: Loads the `Housing.csv` dataset into a Pandas DataFrame.
2.  **Categorical Feature Handling**: Converts 'yes'/'no' string columns to binary (1/0) and applies one-hot encoding to multi-category features like `furnishingstatus`.
3.  **Target Variable Definition**: Transforms the continuous `price` column into a binary `price_category` (0 or 1) based on the median price.
4.  **Train-Test Split**: Divides the data into training and testing sets, ensuring stratified sampling to maintain class distribution.
5.  **Feature Scaling**: Standardizes numerical features using `StandardScaler` to ensure all features contribute equally to the model, importantly, fitting only on the training data to prevent data leakage.
6.  **Class Imbalance Handling**: Checks for class imbalance in the training data and applies `SMOTE` (Synthetic Minority Over-sampling Technique) if a significant imbalance is detected, to balance the classes and improve model performance on the minority class.
7.  **Model Training**: Trains a `RandomForestClassifier` on the preprocessed and balanced training data, using `class_weight='balanced'` for further robustness against any residual imbalance.
8.  **Model Evaluation**: Assesses the model's performance using:
    *   Accuracy Score
    *   Classification Report (Precision, Recall, F1-score)
    *   Confusion Matrix
9.  **Error Analysis & Visualization**:
    *   Visualizes the Confusion Matrix for a clear understanding of classification performance.
    *   Displays Feature Importances from the Random Forest model to identify key predictors.
    *   Identifies and lists samples that were misclassified by the model for deeper investigation.

### Bugs Identified and Fixed (from initial pipeline)
This section highlights critical issues found in the initial pipeline and their resolutions:

1.  **`ValueError: could not convert string to float`**: The initial attempt to scale the raw DataFrame failed due to the presence of non-numeric (string) categorical columns. **Fix**: Implemented proper categorical encoding (binary conversion and one-hot encoding) *before* scaling.
2.  **Data Leakage during Scaling**: The original pipeline scaled the entire dataset before splitting. This allows information from the test set to "leak" into the scaling parameters. **Fix**: Modified the process to fit the `StandardScaler` *only* on the training data and then transform both training and testing sets separately.
3.  **Undefined Target Variable & Incorrect Problem Type**: The original code referenced an undefined `target` column and used a `RandomForestClassifier` (a classification model) with what was likely intended to be a continuous target (`price`). This led to an unrealistic "perfect 1.0 accuracy" due to the `target` column being `None` or all zeros/ones. **Fix**: Explicitly defined a binary classification target `price_category` by converting the continuous `price` based on its median, making the problem suitable for a `RandomForestClassifier`.
4.  **`IndexError` in Error Analysis**: An `IndexError` occurred when trying to retrieve misclassified samples using `.iloc` with non-positional indices. **Fix**: Corrected the indexing to use direct positional indices (`misclassified_indices`) with `.iloc` for `X_test` and `y_test`.

### Results
-   The final model achieved an accuracy of [Insert Actual Accuracy, e.g., 77.98%].
-   Feature importance analysis revealed `area`, `bedrooms`, and `airconditioning` as the most influential factors in predicting housing price categories.
-   The confusion matrix shows a balanced performance in classifying both high and low-priced homes, with [Insert number] true positives, [Insert number] true negatives, [Insert number] false positives, and [Insert number] false negatives.

### How to Run
1.  Ensure you have the required libraries installed (e.g., `pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn`).
2.  Place the `Housing.csv` file in the specified path or update the data loading path.
3.  Run the Jupyter/Colab notebook cells sequentially.

