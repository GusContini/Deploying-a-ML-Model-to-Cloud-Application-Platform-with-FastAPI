# Model Card

## Model Details

- Developed by @GusContini
- Abr24
- version 1.0.0
- Random Forest Binary Classificator:
    - RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=4,
    min_samples_leaf=3,
    n_jobs=-1,
    criterion='gini',
    max_features=0.5,
    oob_score=True,
    random_state=42
    )

## Intended Use

- It's a demo to demonstrate how to create and app using FastAPI to get predictions of High (> 50K) or Low (<= 50K) salaries based on information obtained in census data.

## Evaluation Data

- Data was extracted from here: https://archive.ics.uci.edu/dataset/20/census+income

## Metrics

- {"precision": 0.8, "recall": 0.5, "fbeta": 0.6153846153846154}

## Ethical Considerations

- N/A

## Caveats and Recommendations

- This project was elaborated to and inferecences here made serve only as a demo