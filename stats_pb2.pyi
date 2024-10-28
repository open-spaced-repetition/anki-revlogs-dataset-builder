"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
Copyright: Ankitects Pty Ltd and contributors
License: GNU AGPL, version 3 or later; http://www.gnu.org/licenses/agpl.html
"""

import builtins
import collections.abc
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.internal.enum_type_wrapper
import google.protobuf.message
import typing

@typing.final
class RevlogEntry(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    class _ReviewKind:
        ValueType = typing.NewType("ValueType", builtins.int)
        V: typing_extensions.TypeAlias = ValueType

    class _ReviewKindEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[RevlogEntry._ReviewKind.ValueType], builtins.type):
        DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
        LEARNING: RevlogEntry._ReviewKind.ValueType  # 0
        REVIEW: RevlogEntry._ReviewKind.ValueType  # 1
        RELEARNING: RevlogEntry._ReviewKind.ValueType  # 2
        FILTERED: RevlogEntry._ReviewKind.ValueType  # 3
        MANUAL: RevlogEntry._ReviewKind.ValueType  # 4
        RESCHEDULED: RevlogEntry._ReviewKind.ValueType  # 5

    class ReviewKind(_ReviewKind, metaclass=_ReviewKindEnumTypeWrapper): ...
    LEARNING: RevlogEntry.ReviewKind.ValueType  # 0
    REVIEW: RevlogEntry.ReviewKind.ValueType  # 1
    RELEARNING: RevlogEntry.ReviewKind.ValueType  # 2
    FILTERED: RevlogEntry.ReviewKind.ValueType  # 3
    MANUAL: RevlogEntry.ReviewKind.ValueType  # 4
    RESCHEDULED: RevlogEntry.ReviewKind.ValueType  # 5

    ID_FIELD_NUMBER: builtins.int
    CID_FIELD_NUMBER: builtins.int
    USN_FIELD_NUMBER: builtins.int
    BUTTON_CHOSEN_FIELD_NUMBER: builtins.int
    INTERVAL_FIELD_NUMBER: builtins.int
    LAST_INTERVAL_FIELD_NUMBER: builtins.int
    EASE_FACTOR_FIELD_NUMBER: builtins.int
    TAKEN_MILLIS_FIELD_NUMBER: builtins.int
    REVIEW_KIND_FIELD_NUMBER: builtins.int
    id: builtins.int
    cid: builtins.int
    usn: builtins.int
    button_chosen: builtins.int
    interval: builtins.int
    last_interval: builtins.int
    ease_factor: builtins.int
    taken_millis: builtins.int
    review_kind: global___RevlogEntry.ReviewKind.ValueType
    def __init__(
        self,
        *,
        id: builtins.int = ...,
        cid: builtins.int = ...,
        usn: builtins.int = ...,
        button_chosen: builtins.int = ...,
        interval: builtins.int = ...,
        last_interval: builtins.int = ...,
        ease_factor: builtins.int = ...,
        taken_millis: builtins.int = ...,
        review_kind: global___RevlogEntry.ReviewKind.ValueType = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["button_chosen", b"button_chosen", "cid", b"cid", "ease_factor", b"ease_factor", "id", b"id", "interval", b"interval", "last_interval", b"last_interval", "review_kind", b"review_kind", "taken_millis", b"taken_millis", "usn", b"usn"]) -> None: ...

global___RevlogEntry = RevlogEntry

@typing.final
class CardEntry(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    ID_FIELD_NUMBER: builtins.int
    NOTE_ID_FIELD_NUMBER: builtins.int
    DECK_ID_FIELD_NUMBER: builtins.int
    id: builtins.int
    note_id: builtins.int
    deck_id: builtins.int
    def __init__(
        self,
        *,
        id: builtins.int = ...,
        note_id: builtins.int = ...,
        deck_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["deck_id", b"deck_id", "id", b"id", "note_id", b"note_id"]) -> None: ...

global___CardEntry = CardEntry

@typing.final
class DeckEntry(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    ID_FIELD_NUMBER: builtins.int
    PARENT_ID_FIELD_NUMBER: builtins.int
    PRESET_ID_FIELD_NUMBER: builtins.int
    id: builtins.int
    parent_id: builtins.int
    preset_id: builtins.int
    def __init__(
        self,
        *,
        id: builtins.int = ...,
        parent_id: builtins.int = ...,
        preset_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["id", b"id", "parent_id", b"parent_id", "preset_id", b"preset_id"]) -> None: ...

global___DeckEntry = DeckEntry

@typing.final
class Dataset(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    REVLOGS_FIELD_NUMBER: builtins.int
    CARDS_FIELD_NUMBER: builtins.int
    DECKS_FIELD_NUMBER: builtins.int
    NEXT_DAY_AT_FIELD_NUMBER: builtins.int
    next_day_at: builtins.int
    @property
    def revlogs(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___RevlogEntry]: ...
    @property
    def cards(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___CardEntry]: ...
    @property
    def decks(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___DeckEntry]: ...
    def __init__(
        self,
        *,
        revlogs: collections.abc.Iterable[global___RevlogEntry] | None = ...,
        cards: collections.abc.Iterable[global___CardEntry] | None = ...,
        decks: collections.abc.Iterable[global___DeckEntry] | None = ...,
        next_day_at: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["cards", b"cards", "decks", b"decks", "next_day_at", b"next_day_at", "revlogs", b"revlogs"]) -> None: ...

global___Dataset = Dataset
