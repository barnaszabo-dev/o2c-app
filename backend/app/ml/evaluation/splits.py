"""ONE validation policy for all models: chronological hold-out (train = past, test = future) where all items of one order stay on the same side."""
import pandas as pd

def chrono_group_split(df: pd.DataFrame, test_fraction: float = 0.2):
    first = df.groupby("d_sales_document_base")["Purchase Order Date"].min().sort_values()
    size = df.groupby("d_sales_document_base").size().reindex(first.index)
    need = len(df) * (1 - test_fraction); cum = size.cumsum()
    train_groups = set(first.index[cum <= need]); tr = df["d_sales_document_base"].isin(train_groups)
    cutoff = df.loc[~tr, "Purchase Order Date"].min()
    return df.index[tr], df.index[~tr], cutoff
