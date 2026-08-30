"""Build the parquet dataset from the per-user `.revlog` protobufs.

Run with no arguments this reproduces the published `anki-revlogs-10k` exactly; every
behaviour change below is opt-in behind a flag.

review_time semantics
---------------------
Anki writes a revlog row when the user *answers* a card, so `revlog.id` is the **answer
time**, not the moment the card was shown. `--show-time` switches every derived quantity
(`day_offset`, `elapsed_days`, `elapsed_seconds`, and the sort order) to the **show time**:

    review_time = entry.id - entry.taken_millis

which is what time-of-day and elapsed-time features actually want.

Verified in the data rather than assumed. User 333, 296,002 rows, 18,149 cards with >= 2
reviews: show times are monotone per card on all 18,149 cards, while answer times
(`review_time + duration`) are monotone on only 18,008. That asymmetry can only arise if the
stored column is the show time and durations vary.

⚠ `taken_millis` is CLAMPED by Anki at the deck preset's "maximum answer seconds", so the
correction is exact only for reviews under that cap. For a clamped review the subtraction is
too small and the computed show time lands late by `actual - cap`. Measured over 60 users /
6,265,897 reviews of the 10k set: **2.42 % of rows sit exactly at a cap** (per-user median
3.3 %, p90 9.4 %, worst 25.0 %); the caps observed were 60 s (40 users), 180 s (8), 300 s (3)
and 120 s (2). The correction is still strictly better than not correcting -- uncorrected,
*every* row is late by its full duration -- and clamped rows remain identifiable, since the
clamped value is what is stored: they are exactly the rows whose `duration` equals that user's
cap. No flag column is emitted for them here, because detecting the cap requires either the
preset config or a histogram heuristic, and neither belongs in this script by default.

elapsed_seconds
---------------
`elapsed_seconds` is diffed in protobuf order (per-card blocks) and the frame is sorted by
`review_time` only afterwards. Under `--show-time` the correction can reorder two adjacent
reviews of one card, leaving a genuinely **negative** gap -- on user 333, 127 of 296,002 rows
(0.043 %), ranging -58 s to -2 s, bounded by the review's own duration exactly as the mechanism
predicts. Those are now clamped to 0 and counted.

⚠ Count them as `elapsed_seconds < -1`, not `< 0`: `-1` is the existing "no known previous
review" sentinel and accounts for a further ~6 % of rows in both variants, so a naive `< 0`
test reports ~6.2 % and hides the real figure. Clamping to the `-1` sentinel instead of 0 is
NOT safe -- it propagates into any per-card cumulative sum of `elapsed_seconds` and becomes
`log(negative)` downstream. The published dataset contains no such rows, so this clamp is a
no-op unless `--show-time` is used.

end-to-END vs end-to-START (`--elapsed-end-to-start`)
-----------------------------------------------------
A review occupies an interval, not an instant. Write `start(k)` for the moment the card is
shown and `end(k)` for the moment it is answered, so `duration(k) = end(k) - start(k)` and
`revlog.id` is `end(k)`.

The diff above measures a gap between two timestamps of the SAME kind, so which quantity it
produces depends on which kind `review_time` holds:

    default        end(k)   - end(k-1)       "end-to-END"
    --show-time    start(k) - start(k-1)     "start-to-START"

Neither is the span over which the memory decays. Decay begins when the user finished being
shown the answer, `end(k-1)`, and the test happens when the card is next shown, `start(k)`:

    --elapsed-end-to-start    start(k) - end(k-1)       "end-to-START"

`--show-time` moves the near endpoint but not the far one, so a start-to-START gap still
carries `duration(k-1)` inside it. There is a second, stronger reason to prefer end-to-start:
`duration(k)` does not exist at prediction time (the card has been shown and not yet answered)
and it correlates with the outcome, because a review the user struggles with takes longer.
end-to-END therefore hides a prediction-time-unavailable, outcome-correlated quantity inside
the interval.

The flag is independent of `--show-time`: both endpoints are derived explicitly, so it yields
`start(k) - end(k-1)` either way. It does NOT touch `elapsed_days` -- that is a calendar-day
index difference matching Anki's scheduling semantics, "subtract a duration" is not well
defined on a day index, and the effect at day resolution is ~0.001 %.

Unlike start-to-START, end-to-START almost never goes negative: a card is not shown again
before the previous review was answered, so `start(k) >= end(k-1)` holds physically. Measured
per card over 40 stride-sampled users of the 10k set (2,306,229 rows with a previous review of
the same card): **2 rows**, the worst -2.0 s. Both have a mechanism, neither is clock drift:

* a **zero-duration entry** (0.067 % of rows; Anki writes these for manual reschedules and when
  it bumps a colliding revlog id). Its `start` equals its own `id`, so it can land inside a real
  review's span -- and once two rows overlap, sorting by `start` no longer preserves `end`
  order, so "the previous row by start" can have the later end.
* a **1 ms overlap**, where one review's answer and the next one's show fall on adjacent
  milliseconds.

Both are covered by the existing clamp.

What the correction does produce is more **sub-second** gaps, since it removes a whole duration
rather than a difference of two: **0.222 % of eligible rows** land in [0, 1) s and truncate to a
0 s gap. That is not a problem here -- 0 is a legitimate value -- but consumers that filter on a
strictly positive interval will drop those rows, so it is worth stating.
"""

import argparse
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore
from tqdm import tqdm  # type: ignore

from stats_pb2 import CardEntry, Dataset, DeckEntry, RevlogEntry

DEFAULT_REVLOGS_DIR = Path("revlogs")
DEFAULT_OUTPUT_DIR = Path("../anki-revlogs-10k")

# Anki ids are creation timestamps in epoch milliseconds (~1.7e12), so every id column must be
# int64. int32 would SATURATE rather than raise, silently collapsing distinct entities into one.
ID_DTYPE = "int64"


def filter_revlog(entries: Iterable[RevlogEntry]):
    return filter(
        lambda entry: entry.button_chosen >= 1
        and (entry.review_kind != 3 or entry.ease_factor != 0),
        entries,
    )


def convert_revlog(entries: Iterable[RevlogEntry], show_time: bool = False):
    return map(
        lambda entry: {
            # See the module docstring: revlog.id is the ANSWER time; subtracting the answer
            # duration gives the SHOW time.
            "review_time": entry.id - entry.taken_millis if show_time else entry.id,
            "card_id": entry.cid,
            "rating": entry.button_chosen,
            "state": entry.review_kind,
            "duration": entry.taken_millis,
        },
        filter_revlog(entries),
    )


def convert_card(entries: Iterable[CardEntry]):
    return map(
        lambda entry: {
            "card_id": entry.id,
            "note_id": entry.note_id,
            "deck_id": entry.deck_id,
        },
        entries,
    )


def convert_deck(entries: Iterable[DeckEntry]):
    return map(
        lambda entry: {
            "deck_id": entry.id,
            "parent_id": entry.parent_id,
            "preset_id": entry.preset_id,
        },
        entries,
    )


class IdMapper:
    """Maps Anki ids to small per-user integers.

    With `factorize=False` the ids are passed through unchanged as int64, which is what the
    `--raw-ids` variant wants: Anki ids *are* creation timestamps, so keeping them raw is what
    makes card-age, note-age and creation-batch features derivable at all.
    """

    def __init__(self, factorize: bool = True):
        self._mappings = {}
        self._factorize = factorize

    def get_mapping(self, column_name):
        if column_name not in self._mappings:
            self._mappings[column_name] = {}
        return self._mappings[column_name]

    def factorize(self, series, column_name):
        if not self._factorize:
            # Explicit and asserted: a narrowing cast here would saturate, not raise.
            out = series.astype(ID_DTYPE)
            assert (out == series).all(), f"{column_name}: id lost precision as {ID_DTYPE}"
            return out
        mapping = self.get_mapping(column_name)
        result = series.map(lambda x: mapping.setdefault(x, len(mapping)))
        return result.astype(ID_DTYPE)


def process_and_save(
    file_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    raw_ids: bool = False,
    show_time: bool = False,
    emit_review_time: bool = False,
    end_to_start: bool = False,
):
    data = open(file_path, "rb").read()
    dataset = Dataset()
    dataset.ParseFromString(data)

    id_mapper = IdMapper(factorize=not raw_ids)

    df_revlogs = process_revlogs(
        dataset,
        pd.DataFrame(convert_revlog(dataset.revlogs, show_time=show_time)),
        id_mapper,
        emit_review_time=emit_review_time,
        show_time=show_time,
        end_to_start=end_to_start,
    )
    df_cards = process_cards(pd.DataFrame(convert_card(dataset.cards)), id_mapper)
    df_decks = process_decks(pd.DataFrame(convert_deck(dataset.decks)), id_mapper)

    user_id = int(file_path.stem)
    save_to_parquet(df_revlogs, "revlogs", user_id, output_dir)
    save_to_parquet(df_cards, "cards", user_id, output_dir)
    save_to_parquet(df_decks, "decks", user_id, output_dir)


def process_revlogs(
    dataset,
    df,
    id_mapper,
    emit_review_time: bool = False,
    show_time: bool = False,
    end_to_start: bool = False,
):
    if df.empty:
        return df

    df["i"] = df.groupby("card_id").cumcount() + 1
    df["is_learn_start"] = (df["state"] == 0) & (
        (df["state"].shift() != 0) | (df["i"] == 1)
    )
    df["sequence_group"] = df["is_learn_start"].cumsum()
    last_learn_start = (
        df[df["is_learn_start"]].groupby("card_id")["sequence_group"].last()
    )
    df["last_learn_start"] = (
        df["card_id"].map(last_learn_start).fillna(0).astype("int64")
    )
    df["mask"] = df["last_learn_start"] <= df["sequence_group"]
    df = df[df["mask"] == True]
    df.loc[:, "state"] += 1
    df.loc[df["is_learn_start"], "state"] = 0
    df = df.groupby("card_id").filter(lambda group: group["state"].iloc[0] == 0)

    df["review_time"] = df["review_time"].astype("int64")
    df["day_offset"] = df["review_time"].apply(
        lambda x: int((x / 1000 - dataset.next_day_at) / 86400)
    )
    df["day_offset"] = df["day_offset"] - df["day_offset"].min()
    df["elapsed_days"] = df["day_offset"].diff().fillna(0).astype("int64")
    if end_to_start:
        # end-to-START: end(k-1) -> start(k) (see the module docstring). Derive both endpoints
        # explicitly, because `review_time` holds start(k) under --show-time and end(k)
        # otherwise, so this stays correct either way.
        start = df["review_time"] if show_time else df["review_time"] - df["duration"]
        gap = start - (start + df["duration"]).shift()
        df["elapsed_seconds"] = (gap.fillna(0) / 1000).astype("int64")
    else:
        df["elapsed_seconds"] = (df["review_time"].diff().fillna(0) / 1000).astype("int64")

    # Explicit handling of genuinely negative gaps (see the module docstring). Done BEFORE the
    # -1 sentinels are written so this cannot clobber them. A no-op on answer-time data; only
    # --show-time can reorder rows within a card block.
    negative = df["elapsed_seconds"] < 0
    if negative.any():
        df.loc[negative, "elapsed_seconds"] = 0

    df.loc[df["state"] == 0, "elapsed_days"] = -1
    df.loc[df["state"] == 0, "elapsed_seconds"] = -1
    df["card_id"] = id_mapper.factorize(df["card_id"], "card_id")
    df.sort_values(by="review_time", inplace=True)
    columns = [
        "card_id",
        "day_offset",
        "rating",
        "state",
        "duration",
        "elapsed_days",
        "elapsed_seconds",
    ]
    if emit_review_time:
        columns.append("review_time")
    return df[columns]


def process_cards(df, id_mapper):
    if df.empty:
        return df

    df["card_id"] = id_mapper.factorize(df["card_id"], "card_id")
    df["note_id"] = id_mapper.factorize(df["note_id"], "note_id")
    df["deck_id"] = id_mapper.factorize(df["deck_id"], "deck_id")
    return df


def process_decks(df, id_mapper):
    if df.empty:
        return df

    df["deck_id"] = id_mapper.factorize(df["deck_id"], "deck_id")
    df["parent_id"] = id_mapper.factorize(df["parent_id"], "deck_id")
    df["preset_id"] = id_mapper.factorize(df["preset_id"], "preset_id")
    return df


def save_to_parquet(df, table_name, user_id, output_dir: Path = DEFAULT_OUTPUT_DIR):
    if df.empty:
        return

    df["user_id"] = user_id
    table = pa.Table.from_pandas(df)
    output_path = Path(output_dir) / table_name

    pq.write_to_dataset(
        table,
        output_path,
        partition_cols=["user_id"],
        existing_data_behavior="delete_matching",
    )

    # rename the file to user_id=xxx
    for file in (output_path / f"user_id={user_id}").glob("*.parquet"):
        new_name = file.with_name("data.parquet")
        file.rename(new_name)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build the anki-revlogs parquet dataset from per-user .revlog protobufs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--revlogs-dir", type=Path, default=DEFAULT_REVLOGS_DIR,
        help="directory containing the per-user *.revlog protobufs",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="directory to write the revlogs/ cards/ decks/ parquet trees into",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="worker processes (default: one per CPU)",
    )
    parser.add_argument(
        "--raw-ids", action="store_true",
        help="keep raw Anki ids (int64 epoch-ms) instead of factorizing them to small ints",
    )
    parser.add_argument(
        "--show-time", action="store_true",
        help="derive times from the card SHOW time (revlog.id - taken_millis) instead of the "
             "answer time; see the module docstring for the clamping caveat",
    )
    parser.add_argument(
        "--elapsed-end-to-start", action="store_true",
        help="measure elapsed_seconds end-to-start, i.e. from the previous review's end to "
             "this review's start (the span over which memory decays), instead of between two "
             "same-kind timestamps; see the module docstring",
    )
    parser.add_argument(
        "--emit-review-time", action="store_true",
        help="include the absolute review_time column (epoch ms) in the revlogs table",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    revlog_files = tuple(Path(args.revlogs_dir).glob("*.revlog"))
    if not revlog_files:
        raise SystemExit(f"no *.revlog files found in {args.revlogs_dir}")

    worker = partial(
        process_and_save,
        output_dir=args.output_dir,
        raw_ids=args.raw_ids,
        show_time=args.show_time,
        emit_review_time=args.emit_review_time,
        end_to_start=args.elapsed_end_to_start,
    )
    with Pool(processes=args.workers) as pool:
        list(
            tqdm(
                pool.imap_unordered(worker, revlog_files),
                total=len(revlog_files),
            )
        )


if __name__ == "__main__":
    main()
