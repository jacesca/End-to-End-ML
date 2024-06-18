# End to End project
1. A full cycle project:
- EDA
- Cleaning
- Feature Engineering
- Training
- MLFlow Logging
2. MLFlow --> A Tool for Managing the Machine Learning Lifecycle
3. Feast --> Feature storage
4. Unit Test
5. Logging
6. Data Drift

## Instruction
1. To run the model
```
python end-to-end.py
```

2. To run and review the tracked models in MLFlow

```
mlflow server --host 127.0.0.1 --port 5080
# or
mlflow ui
```
> And then, go to [http://localhost:8080](http://localhost:5080)

3. To execute unittests 
```
python -m unittest -v
# or
python -m unittest -v .\tests\test_model.py
```

## Installing using GitHub
- Fork the project into your GitHub
- Clone it into your dektop
```
git clone https://github.com/jacesca/End-to-End-ML.git
```
- Setup environment (it requires python3)
```
python -m venv venv
source venv/bin/activate  # for Unix-based system
venv\Scripts\activate  # for Windows
```
- Install requirements
```
pip install -r requirements.txt
```

## Extra documentation
- [MLFlow](https://mlflow.org/)
- [Quick Start - MLFlow](https://mlflow.org/docs/latest/getting-started/intro-quickstart/index.html)
- [SHAP documentation](shap.readthedocs.io)
- [Logging levels](https://docs.python.org/3/library/logging.html#levels)
- [MLFlow RestAPI](https://mlflow.org/docs/latest/rest-api.html#create-experiment)
- [Feast QuickStar](https://docs.feast.dev/getting-started/quickstart)
- [PSI Function Development - A practical introduction to Population Stability Index (PSI)](https://www.aporia.com/learn/data-science/practical-introduction-to-population-stability-index-psi/)
- [POPULATION STABILITY INDEX Function Development](https://www.kaggle.com/code/podsyp/population-stability-index)
- [An End-to-End ML Model Monitoring Workflow with NannyML in Python](https://www.datacamp.com/tutorial/model-monitoring-with-nannyml-in-python)
