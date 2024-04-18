from __future__ import annotations
from typing import Optional, Union, Iterable, Callable, List

from time import process_time_ns, time_ns

from .flows import RegressionFlow
from .tasks import OneTargetTask

import pandas as pd
import numpy as np

import pickle
import sys

class Run:

    """
    Docs go here
    """

    def __init__(
        self,
        task: OneTargetTask,
        flow: RegressionFlow,
        name: Optional[str] = None,
        description: Optional[str] = None):

        """
        Docs go here
        """

        self.task = task
        self.flow = flow
        self.description = description

        if name is None:
            self.name = f"Run {self.task.name}_{self.flow.name}"
        else:
            self.name = name

    def run(
        self,
        verbosity: int = 1):

        """
        Docs go here
        """

        X = self.task.features.copy()
        y = self.task.target.copy()

        if self.flow.prior_transformations is not None:
            for (transformation, params) in self.flow.prior_transformations:
                X = transformation(data = X, **params)

        instance_results = {
            # "fold" : [], 
            "instance" : [], 
            "real" : [],
            "prediction" : []}

        profile_results = {
            # "fold" : [],
            "cpu_training_time" : [],
            "cpu_prediction_time" : [],
            "memory_usage" : []}
        
        if self.task.evaluation_method_name == "tt":
            X_train, X_test, y_train, y_test = self.task.splitter
            
            if y_test.values.ndim > 1:
                y_test = y_test.values.flatten()

            t_start = time_ns()

            self.flow.model.fit(X_train, y_train)

            t_middle = time_ns()

            preds = self.flow.model.predict(X_test)

            t_end = time_ns()

            # errors = self.task.evaluation_metric(y_test, preds)

            instance_results["instance"].extend(X_test.index)
            instance_results["real"].extend(y_test)
            instance_results["prediction"].extend(preds)

            profile_results["memory_usage"].append(sys.getsizeof(pickle.dumps(self.flow.model)))
            profile_results["cpu_training_time"].append(t_middle - t_start)
            profile_results["cpu_prediction_time"].append(t_end - t_middle)

            if verbosity > 0:
                print(f"{self.task.evaluation_metric_name} = {self.task.evaluation_metric(y_test, preds)}")

            return RunResult(
                self, 
                instance_results = pd.DataFrame(instance_results), 
                profile_results = pd.DataFrame(profile_results))

        else:

            instance_results["fold"] = []
            profile_results["fold"] = []

            fold_results = {
                "fold" : [],
                self.task.evaluation_metric_name : []}

            for i, (train_indexes, test_indexes) in enumerate(self.task.splitter.split(X, y)):
                # Wtf
                try:
                    aux_model = self.flow.model.__class__(**self.flow.model.get_params())
                except TypeError:
                    aux_model = self.flow.model.__class__(self.flow.model.get_params())

                train_X = X.iloc[train_indexes]
                train_y = y.iloc[train_indexes]

                test_X = X.iloc[test_indexes]
                test_y = y.iloc[test_indexes]

                if test_y.values.ndim > 1:
                    test_y = test_y.values.flatten()

                t_start = time_ns()

                aux_model.fit(train_X, train_y)

                t_middle = time_ns()

                preds = aux_model.predict(test_X)

                t_end = time_ns()
                # errors = self.task.evaluation_metric(test_y, preds)

                fold_error = self.task.evaluation_metric(test_y, preds)

                instance_results["fold"].extend([i+1] * len(test_indexes))
                instance_results["instance"].extend(test_indexes)
                instance_results["real"].extend(test_y)
                instance_results["prediction"].extend(preds)
                # results[self.task.evaluation_metric_name].extend(errors)

                fold_results["fold"].append(i+1)
                fold_results[self.task.evaluation_metric_name].append(fold_error)

                profile_results["fold"].append(i+1)
                profile_results["memory_usage"].append(sys.getsizeof(pickle.dumps(aux_model)))
                profile_results["cpu_training_time"].append(t_middle - t_start)
                profile_results["cpu_prediction_time"].append(t_end - t_middle)

                if verbosity > 0:
                    print(f"CV {i+1} / {self.task.splitter.n_splits} - {self.task.evaluation_metric_name} = {fold_error}")

            for k,v in instance_results.items():
                instance_results[k] = np.array(v)

            for k,v in fold_results.items():
                fold_results[k] = np.array(v)

            return RunResult(
                self, 
                pd.DataFrame(instance_results), 
                pd.DataFrame(fold_results),
                pd.DataFrame(profile_results))
    

class RunResult:

    def __init__(
        self,
        run: Run,
        instance_results: pd.DataFrame,
        fold_results: Optional[pd.DataFrame] = None,
        profile_results: Optional[pd.DataFrame] = None):

        self.run = run
        self.instance_results = instance_results
        self.fold_results = fold_results
        self.profile_results = profile_results

        if fold_results is not None:
            self.score = fold_results[self.run.task.evaluation_metric_name].mean()
        
        else:
            self.score = self.run.task.evaluation_metric(
                self.instance_results["real"].values,
                self.instance_results["prediction"].values)