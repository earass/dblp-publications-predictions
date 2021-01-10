import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from keras.models import Sequential
from keras.layers import Dense
import pickle
from math import sqrt


class Model:
    features_cols = ['JournalId', 'year']
    output_col = 'count'
    data_path = 'engine/data/journal_vol_per_year.csv'

    def __init__(self, model_name):
        self.df = Model.read_data()
        self.eval = pd.DataFrame()
        self.X_train = pd.DataFrame()
        self.X_test = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.y_test = pd.DataFrame()
        self.train_test_eval_split()
        self.model = None
        self.model_name = model_name

    @classmethod
    def read_data(cls):
        return pd.read_csv(Model.data_path)

    def train_test_eval_split(self, cutof_year=2016):
        """ splitting evaluation dataset from 2016 onwards and splitting the rest into train and test"""
        cond = self.df['year'] <= cutof_year
        train_test = self.df[cond]
        self.eval = self.df[~cond]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(train_test[Model.features_cols],
                                                                                train_test[Model.output_col],
                                                                                test_size=0.20)

    def evaluate_model(self):
        X = self.eval[Model.features_cols]
        y_true = self.eval[Model.output_col]
        pred = self.model.predict(X)
        X['y_pred'] = pred
        X['y_true'] = y_true
        print(f"Evaluation for {self.model_name}")
        print(X.head(50))
        error = mean_squared_error(y_true=y_true, y_pred=pred)
        print(f"Root Mean Squared Error for {self.model_name}: {sqrt(error)}")
        return error

    def save_model(self):
        if self.model:
            pickle.dump(self.model, open(f"engine/pickles/{self.model_name}.pickle", 'wb'))

    def load_model(self):
        try:
            return pickle.load(open(f"engine/pickles/{self.model_name}.pickle", 'rb'))
        except:
            raise Exception("Model not found")

    def predict(self, X):
        if not self.model:
            my_model = self.load_model()
            return my_model.predict(X)
        else:
            return self.model.predict(X)


class TrainLinearRegression(Model):
    model_name = 'LinearRegression'

    def __init__(self):
        super().__init__(model_name=TrainLinearRegression.model_name)
        self.model = LinearRegression().fit(X=self.X_train, y=self.y_train)


class TrainDecisionTreeRegressor(Model):
    model_name = 'DecisionTreeRegressor'

    def __init__(self):
        super().__init__(model_name=TrainDecisionTreeRegressor.model_name)
        self.model = DecisionTreeRegressor().fit(X=self.X_train, y=self.y_train)


class TrainNNRegressor(Model):
    model_name = 'NNRegressor'

    def __init__(self):
        super().__init__(model_name=TrainNNRegressor.model_name)
        self.model = self.baseline_model()
        self.model.fit(self.X_train, self.y_train, epochs=20, verbose=1, batch_size=5)

    @staticmethod
    def baseline_model():
        model = Sequential()
        model.add(Dense(2, input_dim=2, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model


class TrainLogisticRegression(Model):
    model_name = 'LogisticRegression'

    def __init__(self):
        super().__init__(model_name=TrainLogisticRegression.model_name)
        self.model = LogisticRegression().fit(X=self.X_train, y=self.y_train)

    def evaluate_model(self, X, y_true):
        pred = self.model.predict(X)
        print(pred)
        error = accuracy_score(y_true=y_true, y_pred=pred)
        print(f"Accuracy: {error}")
        return error


class TrainDecisionTreeClassifier(Model):
    model_name = 'DecisionTreeClassifier'

    def __init__(self):
        super().__init__(model_name=TrainDecisionTreeClassifier.model_name)
        self.model = LogisticRegression().fit(X=self.X_train, y=self.y_train)

    def evaluate_model(self, X, y_true):
        pred = self.model.predict(X)
        print(list(pred))
        error = accuracy_score(y_true=y_true, y_pred=pred)
        print(f"Accuracy: {error}")
        return error
