from engine.model import TrainDecisionTreeRegressor, TrainLinearRegression, TrainNNRegressor


if __name__ == '__main__':

    dtr = TrainDecisionTreeRegressor()
    dtr.evaluate_model()
    # dtr.save_model()
    #
    lr = TrainLinearRegression()
    lr.evaluate_model()
    # lr.save_model()

    # nnr = TrainNNRegressor()
    # nnr.evaluate_model()
    # nnr.save_model()
