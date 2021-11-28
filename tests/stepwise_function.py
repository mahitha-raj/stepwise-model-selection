#!/usr/bin/env python3

import sys
import warnings
import pandas as pd
import statsmodels.api as sm

from copy import deepcopy  # Used to create sentiment word dictionary


warnings.simplefilter(action="ignore", category=FutureWarning)


class StepWise(object):
    def __init__(
        self,
        df,
        yname,
        xnames=None,
        method="stepwise",
        reg=None,
        crit_in=0.1,
        crit_out=0.1,
        x_force=None,
        verbose=False,
        deep=True,
    ):
        if reg == None:
            raise RuntimeError(
                "***Call to stepwise invalid. "
                + "***   Required parameter reg must be set to linear or logistic."
            )
            sys.exit()
        if type(df) != pd.DataFrame:
            raise RuntimeError(
                "***Call to stepwise invalid. "
                + "***   Required data type must be a dataframe."
            )
            sys.exit()
        if df.shape[0] < 2:
            raise RuntimeError(
                "***Call to stepwise invalid. "
                + "***   Required Dataframe has less the 2 observations."
            )
        if type(yname) != str:
            raise RuntimeError(
                "***Call to stepwise invalid. "
                + "***   Required parameter not a string name in DataFrame."
            )
            sys.exit()

        if type(xnames) != type(None):
            if not (all(item in df.columns for item in xnames)):
                raise RuntimeError(
                    "***Call to stepwise invalid. "
                    + "***   xnames are not all in DataFrame."
                )
                sys.exit()
        if method != "stepwise" and method != "forward" and method != "backward":
            raise RuntimeError(
                "***Call to stepwise invalid. " + "***   method is invalid."
            )
            sys.exit()
        if reg != "linear" and reg != "logistic":
            raise RuntimeError(
                "***Call to stepwise invalid. " + "***   reg is invalid."
            )
            sys.exit()
        if type(crit_in) == str:
            if crit_in != "AIC" and crit_in != "BIC":
                raise RuntimeError(
                    "***Call to stepwise invalid. " + "***   crit_in is invalid."
                )
                sys.exit()
        else:
            if type(crit_in) != float:
                raise RuntimeError(
                    "***Call to stepwise invalid. " + "***   crit_in is invalid."
                )
                sys.exit()
            else:
                if crit_in > 1.0 or crit_in < 0.0:
                    raise RuntimeError(
                        "***Call to stepwise invalid. " + "***   crit_in is invalid."
                    )
                    sys.exit()
        if type(crit_out) == str:
            if crit_out != "AIC" and crit_out != "BIC":
                raise RuntimeError(
                    "***Call to stepwise invalid. " + "***   crit_out is invalid."
                )
                sys.exit()
        else:
            if type(crit_out) != float:
                raise RuntimeError(
                    "***Call to stepwise invalid. " + "***   crit_out is invalid."
                )
                sys.exit()
            else:
                if crit_out > 1.0 or crit_out < 0:
                    raise RuntimeError(
                        "***Call to stepwise invalid. " + "***   crit_out is invalid."
                    )
                    sys.exit()
        if type(x_force) != type(None) and not (
            all(item in df.columns for item in x_force)
        ):
            raise RuntimeError(
                "***Call to stepwise invalid. " + "***   x_force is invalid."
            )
            sys.exit()
        if deep == True:
            self.df_copy = deepcopy(df)
        else:
            self.df_copy = df

        # string - column name in df for y
        self.yname = yname
        # None or string = list of column names in df for X var.
        if type(xnames) != type(None):
            self.xnames = xnames  # list of strings (col names)
        else:
            self.xnames = list(set(df.columns) - set([yname]))
        # string - "stepwise", "backward" or "forward"
        self.method = method  # string
        # string - "linear" or "logistic"
        self.reg = reg  # string
        # string = "AIC" or "BIC", or p=[0,1]
        if type(crit_in) == str or type(crit_out) == str:
            warnings.warn(
                "\n***Call to stepwise invalid: "
                + " crit_in and crit_out must be a number between 0 and 1."
            )
            self.crit_in = 0.1
            self.crit_out = 0.1
        else:
            self.crit_in = crit_in  # float
            self.crit_out = crit_out  # float
        # [] of string = list of column names in df forced into model
        if type(x_force) != type(None):
            self.x_force = x_force  # list of strings (col names)
        else:
            self.x_force = []
        # True or False, control display of steps selected
        self.verbose = verbose
        # initialized list of selected columns in df
        self.selected_ = []

        return

    # **************************************************************************

    def stepwise_(self):
        initial_list = []
        included = initial_list
        if self.crit_out < self.crit_in:
            warnings.warn(
                "\n***Call to stepwise invalid: "
                + "crit_out < crit_in, setting crit_out to crit_in"
            )
            self.crit_out = self.crit_in

        X = self.df_copy[self.xnames]
        y = self.df_copy[self.yname]
        while True:
            changed = False
            # forward step
            excluded = list(set(X.columns) - set(included))
            new_pval = pd.Series(index=excluded)
            if self.reg == "linear":
                for new_column in excluded:
                    model = sm.OLS(
                        y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))
                    ).fit()
                    new_pval[new_column] = model.pvalues[new_column]
            else:
                for new_column in excluded:
                    model = sm.Logit(
                        y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))
                    ).fit(disp=False)
                    new_pval[new_column] = model.pvalues[new_column]
            best_pval = new_pval.min()
            if best_pval < self.crit_in:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed = True
                if self.verbose:
                    print(
                        "Add  {:30} with p-value {:.6}".format(best_feature, best_pval)
                    )
            # backward step
            if self.reg == "linear":
                model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
            else:
                model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included]))).fit(
                    disp=False
                )
            pvalues = model.pvalues.iloc[1:]
            worst_pval = pvalues.max()
            if worst_pval > self.crit_out:
                worst_feature = pvalues.idxmax()
                included.remove(worst_feature)
                changed = True
                if self.verbose:
                    print(
                        "Remove {:30} with p-value {:.6}".format(
                            worst_feature, worst_pval
                        )
                    )
            if not changed:
                break
        return included

    # **************************************************************************
    def forward_(self):
        initial_list = []
        included = list(initial_list)
        X = self.df_copy[self.xnames]
        y = self.df_copy[self.yname]
        while True:
            changed = False
            excluded = list(set(X.columns) - set(included))
            new_pval = pd.Series(index=excluded)
            if self.reg == "linear":
                for new_column in excluded:
                    model = sm.OLS(
                        y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))
                    )
                    results = model.fit()
                    new_pval[new_column] = results.pvalues[new_column]
            else:
                for new_column in excluded:
                    model = sm.Logit(
                        y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))
                    ).fit(disp=False)
                    new_pval[new_column] = model.pvalues[new_column]
            best_pval = new_pval.min()
            if best_pval < self.crit_in:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed = True
                if self.verbose:
                    print(
                        "Add  {:30} with p-value {:.6}".format(best_feature, best_pval)
                    )

            if not changed:
                break

        return included

    # **************************************************************************

    def backward_(self):
        included = list(self.xnames)
        X = self.df_copy[included]
        y = self.df_copy[self.yname]
        while True:
            changed = False
            new_pval = pd.Series(index=included)
            if self.reg == "linear":
                model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
            else:
                model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included]))).fit(
                    disp=False
                )

            for new_column in included:
                new_pval[new_column] = model.pvalues[new_column]
            worst_pval = new_pval.max()
            if worst_pval > self.crit_out:
                worst_feature = new_pval.idxmax()
                included.remove(worst_feature)
                changed = True
                if self.verbose:
                    print(
                        "Remove  {:30} with p-value {:.6}".format(
                            worst_feature, worst_pval
                        )
                    )
            if not changed:
                break
        return included

    # **************************************************************************

    def fit_transform(self):
        if self.method == "stepwise":
            self.selected_ = self.stepwise_()
        else:
            if self.method == "forward":
                self.selected_ = self.forward_()
            else:
                self.selected_ = self.backward_()
        return self.selected_


# **************************************************************************
