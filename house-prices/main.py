# Disclaimer:
# Code adapted from 'Serigno' - https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard

import warnings


def ignore_warn(*args, **kwargs):
    pass


warnings.warn = ignore_warn  # ignore annoying warning (from sklearn and seaborn)


if __name__ == "__main__":

    import preprocessing
    import modelling_tpto_elasticNETCV
