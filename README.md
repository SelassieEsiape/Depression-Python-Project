
# Student Depression Prediction and Analysis

This repository hosts a comprehensive data analytics and machine learning project that explores the prediction of student depression using regression models and visualizations. The project integrates Python, statistical techniques, and machine learning to analyze factors contributing to student depression and offers predictive insights based on user input.

## Project Overview
Depression among students is a critical issue that affects academic performance and overall well-being. This project leverages regression analysis, logistic regression, and a correlation matrix to identify patterns and predictive factors related to student depression. The analysis is supported by a dataset containing anonymized information about students' academic, financial, and lifestyle factors.

### Features
- **Regression Analysis**: Using multiple linear regression to identify significant predictors of student depression.
- **Correlation Matrix**: Examining relationships between predictors to highlight key influencing factors.
- **Logistic Regression**: Building a predictive model to classify students based on their risk of depression.
- **Custom Predictions**: Providing predictions for new data inputs, enabling real-world applications.
- **Power BI Integration**: Exporting correlation data for enhanced visualization and storytelling.

## Key Accomplishments
This project reflects my expertise in:
- Data preprocessing, regression analysis, and machine learning using Python libraries like `pandas`, `statsmodels`, and `scikit-learn`.
- Designing interactive and informative dashboards in **Power BI**, showcasing my data visualization skills.
- Utilizing statistical techniques to derive actionable insights, aiding decision-makers in identifying risk factors and preventive strategies.
- Developing scalable and maintainable code to support analysis and predictive modeling.

## Repository Contents
- **`Depression_Student_Dataset.csv`**: Input dataset for analysis.
- **`main.py`**: Python script implementing the project.
- **`Correlation_Data.csv`**: Exported correlation data for Power BI.
- **README.md**: Documentation explaining the project.

## How to Use
### Prerequisites
- Python 3.x
- Required Python libraries: `pandas`, `numpy`, `statsmodels`, `scikit-learn`, `matplotlib`
- Power BI (optional, for dashboard visualization)

### Steps to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/YourUsername/Student-Depression-Prediction.git
   cd Student-Depression-Prediction
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy statsmodels scikit-learn matplotlib
   ```
3. Place the dataset (`Depression_Student_Dataset.csv`) in the project directory.
4. Run the script:
   ```bash
   python main.py
   ```
5. To use the predictive model, modify the `input` variable in the script to match your custom data.

### Example Input
```python
input = [22, 4.0, 3.0, 6, 4, 1, 1, 0, 0, 1, 0, 1, 1]
```
This corresponds to the following attributes:
- Age: 22
- Academic Pressure: 4 (High)
- Study Satisfaction: 3 (Moderate)
- Study Hours: 6
- Financial Stress: 4 (High)
- Gender: Male
- Sleep Duration: 7-8 hours
- Dietary Habits: Moderate
- Suicidal Thoughts: Yes
- Family History of Mental Illness: Yes

### Results
The script will display:
- Predicted depression status (Yes/No)
- Prediction probability (confidence score)

## Power BI Dashboard
The exported correlation data (`Correlation_Data.csv`) can be loaded into Power BI to create interactive dashboards that visualize relationships between factors and student depression trends.

## Future Enhancements
- Deploying the model as a web app for wider accessibility.
- Incorporating larger datasets to improve model accuracy.
- Expanding features to include real-time data streaming and predictive analytics.

## Contact
For any questions or collaboration opportunities, feel free to reach out:
- **Website**: [selassieesiape.com](https://selassieesiape.com)
- **Email**: selassie@esiape.com
