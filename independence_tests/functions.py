import scipy
import scipy.stats


def resit(X, Y, Z, sklearn_model):
    """
    Independently model X and Y as a
    function of Z using models that follow
    the sklearn fit and predict api.

    Predict both X and Y and retrieve residuals.

    Run unconditional independence test between
    the residuals from the X model and the
    residuals from the Y model.

    http://auai.org/uai2017/proceedings/papers/250.pdf
    """
    if len(Z) == 0:
        # unconditional independence test
        return scipy.stats.ttest_ind(X, Y).pvalue
    else:
        model_X = sklearn_model
        model_X.fit(Z, X)
        X_hat = model_X.predict(Z)
        model_Y = sklearn_model
        model_X.fit(Z, X)
        model_Y.fit(Z, Y)
        Y_hat = model_Y.predict(Z)
        X_res = X_hat - X
        Y_res = Y_hat - Y
        return scipy.stats.ttest_ind(X_res, Y_res).pvalue
