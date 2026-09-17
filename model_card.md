# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a binary classifier that predicts whether a person's annual income is above $50,000 based on census attributes. It was developed as part of the Udacity "Deploying a Scalable ML Pipeline with FastAPI" project.

The model is a scikit-learn `RandomForestClassifier` (scikit-learn 1.5.1, Python 3.10) with 100 trees, a maximum depth of 15, a minimum of 2 samples per leaf, and `random_state=42` so that training is reproducible. The depth and leaf limits were chosen to keep the saved model small enough to commit to GitHub. They reduce the model file from about 74 MB to about 9.5 MB while changing the F1 score by less than 0.01.

The eight categorical features are one-hot encoded with a `OneHotEncoder` that ignores categories it did not see during training, and the six numeric features are used without scaling. The label is binarized so that `<=50K` is 0 and `>50K` is 1. The trained model is saved to `model/model.pkl` and the fitted encoder to `model/encoder.pkl`.

## Intended Use

The model is intended for educational purposes, to demonstrate how to train, evaluate, test, and deploy a machine learning model behind a REST API with continuous integration. It can be used to explore how a classifier's performance varies across demographic groups.

The model is not intended for real-world decisions about individuals, such as hiring, lending, insurance, housing, or eligibility for public services. It was trained on data from 1994 and shows uneven performance across groups, so it should not be used in any setting where its predictions could affect people.

## Training Data

The model was trained on the Census Income ("Adult") dataset from the UCI Machine Learning Repository, which was extracted from the 1994 U.S. Census database. The full dataset contains 32,561 rows with 14 features and one label (`salary`).

The features include six numeric columns (`age`, `fnlgt`, `education-num`, `capital-gain`, `capital-loss`, and `hours-per-week`) and eight categorical columns (`workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, and `native-country`). About 75.9% of rows are labeled `<=50K` and 24.1% are labeled `>50K`.

Missing values in the dataset are recorded as the string `?` in `workclass`, `occupation`, and `native-country`. These rows were kept, and `?` is treated as its own category. The data was split into 80% training data (26,048 rows) using scikit-learn's `train_test_split` with `random_state=42`, stratified on the label so that both sets keep the same class balance.

## Evaluation Data

The evaluation data is the remaining 20% of the dataset (6,513 rows) from the same stratified split. It was processed with the encoder and label binarizer that were fitted on the training data, so no information from the test set was used during training.

In addition to the overall evaluation, the model was evaluated on slices of the test set, with one slice for every unique value of each of the eight categorical features, for 101 slices in total. The full slice results are saved in `slice_output.txt`.

## Metrics

The model was evaluated with precision, recall, and F1 score, with `>50K` as the positive class. Accuracy was not used as the main metric because the classes are imbalanced, and a model that always predicted `<=50K` would still be about 76% accurate.

On the test set, the model achieved a precision of 0.8009, a recall of 0.5874, and an F1 score of 0.6777. This means that when the model predicts an income above $50K it is correct about 80% of the time, but it identifies only about 59% of the people who actually earn above $50K.

The slice analysis shows that performance varies across groups:

| Slice | Count | Precision | Recall | F1 |
|---|---|---|---|---|
| sex: Male | 4,355 | 0.7897 | 0.6017 | 0.6830 |
| sex: Female | 2,158 | 0.8803 | 0.5102 | 0.6460 |
| race: White | 5,533 | 0.8069 | 0.5929 | 0.6836 |
| race: Black | 662 | 0.8000 | 0.4731 | 0.5946 |
| race: Asian-Pac-Islander | 200 | 0.6667 | 0.6538 | 0.6602 |
| race: Amer-Indian-Eskimo | 73 | 0.7500 | 0.3333 | 0.4615 |
| education: Bachelors | 1,096 | 0.7470 | 0.8650 | 0.8017 |
| education: Masters | 318 | 0.8426 | 0.9121 | 0.8760 |
| education: HS-grad | 2,120 | 0.9506 | 0.2271 | 0.3667 |
| relationship: Own-child | 1,032 | 0.6667 | 0.1333 | 0.2222 |

## Ethical Considerations

The dataset includes sensitive attributes such as race, sex, marital status, and native country, and the model uses all of them as features. The underlying data reflects income inequality that existed in 1994, including a much higher rate of incomes above $50K among men (30.6%) than women (10.9%) and large differences in that rate across racial groups. A model trained on this data can learn and repeat those patterns.

The slice results show that the model's recall is lower for women (0.51) than for men (0.60), and lower for Black (0.47) and American Indian or Eskimo (0.33) individuals than for White individuals (0.59). In practice, this means the model is more likely to miss higher earners in these groups. The model also performs much worse for people without a college degree, such as high school graduates, where recall is only 0.23.

The `fnlgt` feature is a census sampling weight rather than a personal attribute, and including it as a predictor has no clear real-world meaning. Because of these issues, the model should not be used to make or support decisions about real people.

## Caveats and Recommendations

The data is more than 30 years old, and incomes, jobs, and demographics have changed a great deal since then. The $50K threshold is not adjusted for inflation, so the model should not be assumed to reflect current conditions.

Many slices are too small for their metrics to be reliable. 44 of the 101 slices have fewer than 30 rows in the test set, and some countries appear only a handful of times. When a slice has no positive examples or the model makes no positive predictions, the metrics fall back to a value of 1.0 because of the `zero_division=1` setting. This is why some small slices show perfect or unusual scores, such as a precision of 1.0 with a recall of 0.0.

To improve the model, future work could tune the classification threshold or use class weights to raise recall, try cross-validation and hyperparameter tuning, remove or test the effect of removing sensitive features and `fnlgt`, and apply fairness metrics and mitigation techniques. The model should be retrained and re-evaluated on more recent data before being used for anything beyond education.
