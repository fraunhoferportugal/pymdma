import pandas as pd

# from pycaret.regression import RegressionExperiment


class RegressionML:
    def __init__(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame = None,
        target_col: str = "",
        **kwargs,
    ) -> None:

        raise NotImplementedError("RegressionML is not implemented yet")

        # data to feed models
        self.train_data = train_data

        # data to test models
        self.test_data = test_data

        # target column (attribute to predict)
        self.target = target_col

        # experiment
        self.exp = RegressionExperiment()

        # models to include
        self.m_inc = [
            "lr",
            "ridge",
            "huber",
            "svm",
            "knn",
            "rf",
            "et",
            "ada",
            "gbr",
        ]

    def prepare_run_(self, **kwargs):
        # experiment
        self.exp.setup(
            data=self.train_data,
            test_data=self.test_data,
            target=self.target,
            index=False,
            verbose=False,
            system_log=False,
            html=False,
            log_experiment=False,
            **kwargs,
        )

    def do_run_(self, **kwargs):
        return self.exp.compare_models(include=self.m_inc, **kwargs)
