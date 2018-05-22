def test_conditional_independence(self, X, Y, Zs=[]):
    """Chi-square conditional independence test.
    Tests the null hypothesis that X is independent from Y given Zs.

    This is done by comparing the observed frequencies with the expected
    frequencies if X,Y were conditionally independent, using a chisquare
    deviance statistic. The expected frequencies given independence are
    `P(X,Y,Zs) = P(X|Zs)*P(Y|Zs)*P(Zs)`. The latter term can be computed
    as `P(X,Zs)*P(Y,Zs)/P(Zs).

    Parameters
    ----------
    X: int, string, hashable object
        A variable name contained in the data set
    Y: int, string, hashable object
        A variable name contained in the data set, different from X
    Zs: list of variable names
        A list of variable names contained in the data set, different from X and Y.
        This is the separating set that (potentially) makes X and Y independent.
        Default: []

    Returns
    -------
    chi2: float
        The chi2 test statistic.
    p_value: float
        The p_value, i.e. the probability of observing the computed chi2
        statistic (or an even higher value), given the null hypothesis
        that X _|_ Y | Zs.
    sufficient_data: bool
        A flag that indicates if the sample size is considered sufficient.
        As in [4], require at least 5 samples per parameter (on average).
        That is, the size of the data set must be greater than
        `5 * (c(X) - 1) * (c(Y) - 1) * prod([c(Z) for Z in Zs])`
        (c() denotes the variable cardinality).


    References
    ----------
    [1] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
    Section 18.2.2.3 (page 789)
    [2] Neapolitan, Learning Bayesian Networks, Section 10.3 (page 600ff)
        http://www.cs.technion.ac.il/~dang/books/Learning%20Bayesian%20Networks(Neapolitan,%20Richard).pdf
    [3] Chi-square test https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test#Test_of_independence
    [4] Tsamardinos et al., The max-min hill-climbing BN structure learning algorithm, 2005, Section 4

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pgmpy.estimators import ConstraintBasedEstimator
    >>> data = pd.DataFrame(np.random.randint(0, 2, size=(50000, 4)), columns=list('ABCD'))
    >>> data['E'] = data['A'] + data['B'] + data['C']
    >>> c = ConstraintBasedEstimator(data)
    >>> print(c.test_conditional_independence('A', 'C'))  # independent
    (0.95035644482050263, 0.8132617142699442, True)
    >>> print(c.test_conditional_independence('A', 'B', 'D'))  # independent
    (5.5227461320130899, 0.59644169242588885, True)
    >>> print(c.test_conditional_independence('A', 'B', ['D', 'E']))  # dependent
    (9192.5172226063387, 0.0, True)
    """

    if isinstance(Zs, (frozenset, list, set, tuple,)):
        Zs = list(Zs)
    else:
        Zs = [Zs]

    num_params = ((len(self.state_names[X])-1) *
                  (len(self.state_names[Y])-1) *
                  np.prod([len(self.state_names[Z]) for Z in Zs]))
    sufficient_data = len(self.data) >= num_params * 5
    if not sufficient_data:
        warn("Insufficient data for testing {0} _|_ {1} | {2}. ".format(X, Y, Zs) +
             "At least {0} samples recommended, {1} present.".format(5 * num_params, len(self.data)))

    # compute actual frequency/state_count table:
    # = P(X,Y,Zs)
    XYZ_state_counts = pd.crosstab(index=self.data[X],
                                   columns=[self.data[Y]] + [self.data[Z] for Z in Zs])
    # reindex to add missing rows & columns (if some values don't appear in data)
    row_index = self.state_names[X]
    column_index = pd.MultiIndex.from_product(
                        [self.state_names[Y]] + [self.state_names[Z] for Z in Zs], names=[Y]+Zs)
    XYZ_state_counts = XYZ_state_counts.reindex(index=row_index,    columns=column_index).fillna(0)

    # compute the expected frequency/state_count table if X _|_ Y | Zs:
    # = P(X|Zs)*P(Y|Zs)*P(Zs) = P(X,Zs)*P(Y,Zs)/P(Zs)
    if Zs:
        XZ_state_counts = XYZ_state_counts.sum(axis=1, level=Zs)  # marginalize out Y
        YZ_state_counts = XYZ_state_counts.sum().unstack(Zs)      # marginalize out X
    else:
        XZ_state_counts = XYZ_state_counts.sum(axis=1)
        YZ_state_counts = XYZ_state_counts.sum()
    Z_state_counts = YZ_state_counts.sum()  # marginalize out both

    XYZ_expected = pd.DataFrame(index=XYZ_state_counts.index, columns=XYZ_state_counts.columns)
    for X_val in XYZ_expected.index:
        if Zs:
            for Y_val in XYZ_expected.columns.levels[0]:
                XYZ_expected.loc[X_val, Y_val] = (XZ_state_counts.loc[X_val] *
                                                  YZ_state_counts.loc[Y_val] /
                                                  Z_state_counts).values
        else:
            for Y_val in XYZ_expected.columns:
                XYZ_expected.loc[X_val, Y_val] = (XZ_state_counts.loc[X_val] *
                                                  YZ_state_counts.loc[Y_val] /
                                                  float(Z_state_counts))

    observed = XYZ_state_counts.values.flatten()
    expected = XYZ_expected.fillna(0).values.flatten()
    # remove elements where the expected value is 0;
    # this also corrects the degrees of freedom for chisquare
    observed, expected = zip(*((o, e) for o, e in zip(observed, expected) if not e == 0))

    chi2, significance_level = chisquare(observed, expected)

    return (chi2, significance_level, sufficient_data)
