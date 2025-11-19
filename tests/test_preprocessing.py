import numpy as np
import pandas as pd
from src.preprocessing.gse48113 import quantile_normalize


def test_quantile_normalize_equalizes_distributions():
    rng = np.random.RandomState(0)
    df = pd.DataFrame(rng.exponential(scale=[10, 100, 1000], size=(100, 3)))
    df_qn = quantile_normalize(df)
    # shapes preserved
    assert df.shape == df_qn.shape
    # columns should share the same sorted values (quantile normalization property)
    sorted_cols = [np.sort(df_qn.iloc[:, i].values) for i in range(df_qn.shape[1])]
    for i in range(1, len(sorted_cols)):
        assert np.allclose(sorted_cols[0], sorted_cols[i])
