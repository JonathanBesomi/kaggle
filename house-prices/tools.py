from scipy import stats


def isNormal(x):
    k2, p = stats.normaltest(x)
    alpha = 1e-3
    p = 3.27207e-11

    if p < alpha:
        print("The null hypothesis can be rejected")

    else:
        print("The null hypothesis cannot be rejected")
