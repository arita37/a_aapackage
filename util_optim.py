


"""
import ...

# Define an objective function to be minimized.
def objective(trial):

    # Invoke suggest methods of a Trial object to generate hyperparameters.
    classifier_name = trial.suggest_categorical('classifier', ['SVC', 'RandomForest'])
    if classifier_name == 'SVC':
        svc_c = trial.suggest_loguniform('svc_c', 1e-10, 1e10)
        classifier_obj = sklearn.svm.SVC(C=svc_c)
    else:
        rf_max_depth = trial.suggest_int('rf_max_depth', 2, 32)
        classifier_obj = sklearn.ensemble.RandomForestClassifier(max_depth=rf_max_depth)

    iris = sklearn.datasets.load_iris()
    x, y = iris.data, iris.target
    score = sklearn.model_selection.cross_val_score(classifier_obj, x, y)
    accuracy = score.mean()

    return 1.0 - accuracy  # A objective value linked with the Trial object.

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=100)  # Invoke optimization of the objective function.
Installation
To install Optuna, use pip as follows:

$ pip install optuna

"""




