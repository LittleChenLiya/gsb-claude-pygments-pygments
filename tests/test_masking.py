"""
Tests for pygments.masking (token stream masking without re-lexing).

Run only these tests with:

    python -m pytest tests/test_masking.py -q
"""

import pytest

from pygments.lexers import PythonLexer
from pygments.masking import (
    MaskError, MaskResult, MaskedRegion, mask_token_stream,
)
from pygments.token import Name, Operator, String, Text


def join(tokens):
    return ''.join(value for _, value in tokens)


# A small hand-built token stream: source 'ab"cdef"gh' (length 10)
#   Text   'ab'   offsets 0..2
#   String '"cd'  offsets 2..5
#   String 'ef"'  offsets 5..8
#   Text   'gh'   offsets 8..10
TOKENS = [
    (Text, 'ab'),
    (String, '"cd'),
    (String, 'ef"'),
    (Text, 'gh'),
]
SOURCE = join(TOKENS)


def test_mask_spanning_two_tokens_only_touches_covered_text():
    # Secret 'cdef' spans the two String tokens (offsets 3..7).
    result = mask_token_stream(TOKENS, [(3, 7)], replacement='***')
    assert result.tokens == [
        (Text, 'ab'),
        (String, '"***'),
        (String, '***"'),
        (Text, 'gh'),
    ]
    # Token types of the outside tokens are preserved.
    assert [t for t, _ in result.tokens] == [Text, String, String, Text]
    assert result.text == 'ab"******"gh'
    # The masked output region maps back to the original source offsets.
    assert result.mapping == [MaskedRegion(3, 9, 3, 7)]
    assert result.text[3:9] == '******'
    assert SOURCE[3:7] == 'cdef'


def test_mapping_offsets_after_length_change_still_point_at_source():
    # Replacement is longer than the masked source text; offsets of text
    # after the masked region shift in the output, but the mapping still
    # refers to the original source offsets.
    result = mask_token_stream(TOKENS, [(0, 2), (8, 10)], replacement='<x>')
    assert result.text == '<x>"cdef"<x>'
    assert result.mapping == [
        MaskedRegion(0, 3, 0, 2),
        MaskedRegion(9, 12, 8, 10),
    ]
    # Everything after the first region is shifted by +1 in the output,
    # while src offsets stay anchored to the original source.
    assert result.text[3:9] == SOURCE[2:8]


def test_shorter_replacement_and_multiple_ranges():
    result = mask_token_stream(TOKENS, [(1, 4), (7, 10)], replacement='')
    assert result.text == 'adef'
    assert result.mapping == [
        MaskedRegion(1, 1, 1, 4),
        MaskedRegion(4, 4, 7, 10),
    ]


def test_empty_range_is_pure_insertion():
    # Empty range: masks zero source characters, inserts the replacement.
    result = mask_token_stream(TOKENS, [(4, 4)], replacement='##')
    assert result.text == 'ab"c##def"gh'
    assert result.mapping == [MaskedRegion(4, 6, 4, 4)]
    # Nothing was removed from the source.
    assert result.text[:4] + result.text[6:] == SOURCE


def test_empty_range_at_token_boundary_and_at_end():
    # Insertion exactly at a token boundary attaches to the next token.
    result = mask_token_stream(TOKENS, [(2, 2)], replacement='!')
    assert result.tokens == [
        (Text, 'ab'),
        (String, '!"cd'),
        (String, 'ef"'),
        (Text, 'gh'),
    ]
    # Insertion at the very end of the source extends the last token.
    result = mask_token_stream(TOKENS, [(len(SOURCE),) * 2], replacement='?')
    assert result.tokens[-1] == (Text, 'gh?')
    assert result.mapping == [MaskedRegion(len(SOURCE), len(SOURCE) + 1,
                                           len(SOURCE), len(SOURCE))]


def test_empty_range_at_empty_stream():
    result = mask_token_stream([], [(0, 0)], replacement='x')
    assert result.tokens == [(Text, 'x')]
    assert result.text == 'x'
    assert result.mapping == [MaskedRegion(0, 1, 0, 0)]


def test_ranges_are_sorted_before_application():
    result = mask_token_stream(TOKENS, [(9, 10), (0, 1)], replacement='*')
    assert result.text == '*b"cdef"g*'
    # Mapping follows the sorted range order.
    assert result.mapping == [
        MaskedRegion(0, 1, 0, 1),
        MaskedRegion(9, 10, 9, 10),
    ]


def test_original_token_stream_is_not_mutated():
    tokens = list(TOKENS)
    snapshot = list(tokens)
    mask_token_stream(tokens, [(0, 10)], replacement='x')
    assert tokens == snapshot
    assert join(tokens) == SOURCE


def test_no_ranges_returns_equal_stream():
    result = mask_token_stream(TOKENS, [])
    assert result.tokens == TOKENS
    assert result.text == SOURCE
    assert result.mapping == []
    assert isinstance(result, MaskResult)


@pytest.mark.parametrize('ranges', [
    [(1, 4), (3, 6)],    # straddling overlap
    [(1, 5), (2, 3)],    # containment
    [(2, 4), (2, 4)],    # identical ranges
    [(3, 5), (1, 4)],    # overlap after sorting
])
def test_overlapping_ranges_are_rejected(ranges):
    with pytest.raises(MaskError, match='overlap'):
        mask_token_stream(TOKENS, ranges)


@pytest.mark.parametrize('ranges', [
    [(-1, 2)],                    # negative start
    [(0, len(SOURCE) + 1)],       # past the end
    [(5, 4)],                     # start > end
    [(0, 2, 3)],                  # not a pair
    [('a', 2)],                   # non-integer offsets
])
def test_out_of_bounds_ranges_are_rejected(ranges):
    with pytest.raises(MaskError):
        mask_token_stream(TOKENS, ranges)


def test_surrogate_pair_must_not_be_split():
    # 'a' + high/low surrogate pair + 'b' (offsets: a=0, pair=1..3, b=3).
    # Python strings are sequences of code points, so build the pair from
    # explicit lone surrogates to model a UTF-16-style split hazard.
    tokens = [(Text, 'a\ud83d\ude00b')]
    assert len(join(tokens)) == 4
    for bad in [(1, 2), (2, 3), (0, 2), (2, 4)]:
        with pytest.raises(MaskError, match='surrogate'):
            mask_token_stream(tokens, [bad])
    # Masking the whole pair is fine.
    result = mask_token_stream(tokens, [(1, 3)], replacement='?')
    assert result.text == 'a?b'
    assert result.mapping == [MaskedRegion(1, 2, 1, 3)]


def test_real_lexer_token_stream_example():
    # Token before/after example on a real lexer stream.  The lexer runs
    # once to produce the input stream; masking itself does not re-lex.
    source = 'token = "secret123"' + chr(10)
    tokens = list(PythonLexer().get_tokens(source))
    start = source.index('secret123')
    end = start + len('secret123')

    result = mask_token_stream(tokens, [(start, end)],
                               replacement='<redacted>')

    # The masked text replaces exactly the secret.
    assert result.text == 'token = "<redacted>"' + chr(10)
    # Only String tokens changed; the surrounding tokens are untouched
    # and keep their original types and (source-side) offsets.
    before = [(t, v) for t, v in tokens]
    after = result.tokens
    assert len(before) == len(after)
    for (t0, v0), (t1, v1) in zip(before, after):
        assert t0 is t1
        if 'secret123' not in v0:
            assert v0 == v1
    # The mapping points back at the original source offsets.
    assert result.mapping == [
        MaskedRegion(start, start + len('<redacted>'), start, end)]
    assert result.text[start:start + len('<redacted>')] == '<redacted>'
    # Name/Operator tokens around the secret are preserved verbatim.
    assert (Name, 'token') in result.tokens
    assert (Operator, '=') in result.tokens
