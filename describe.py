# %%
import pandas as pd  # type: ignore

# %%
df_revlogs_total = pd.read_parquet("../anki-revlogs-10k/revlogs")
print(df_revlogs_total.shape)
df_revlogs_total.describe()

# %%
df_cards_total = pd.read_parquet("../anki-revlogs-10k/cards")
print(df_cards_total.shape)
df_cards_total.describe()

# %%
df_decks_total = pd.read_parquet("../anki-revlogs-10k/decks")
print(df_decks_total.shape)
df_decks_total.describe()

# %%
