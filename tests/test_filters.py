"""
    Tests for the filter module
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Local commands to run these tests::

        # run all filter tests
        pytest tests/test_filters.py -v

        # run a specific test
        pytest tests/test_filters.py::TestTokenStreamMasker::test_mask_spans_two_tokens -v

        # run with short traceback
        pytest tests/test_filters.py --tb=short

    :copyright: Copyright 2006-present by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

import pytest

from pygments.filters import TokenStreamMasker
from pygments.token import Token


def mask(ranges, tokens, replacement='***', **kwargs):
    """Apply `TokenStreamMasker` and return ``(output_tokens, mapping)``."""
    masker = TokenStreamMasker(ranges=ranges, replacement=replacement, **kwargs)
    out = list(masker.filter(None, tokens))
    return out, masker.mapping


class TestTokenStreamMasker:

    def test_no_ranges_returns_stream_unchanged(self):
        # source: 'def foo(): pass'
        tokens = [(Token.Keyword, 'def'), (Token.Text, ' '),
                  (Token.Name.Function, 'foo'), (Token.Text, '('),
                  (Token.Text, ')'), (Token.Text, ' '),
                  (Token.Keyword, 'pass')]
        out, mapping = mask([], tokens)
        assert out == tokens
        # mapping covers the whole source contiguously
        assert mapping == [
            (0, 3, 0, 3, Token.Keyword),
            (3, 4, 3, 4, Token.Text),
            (4, 7, 4, 7, Token.Name.Function),
            (7, 8, 7, 8, Token.Text),
            (8, 9, 8, 9, Token.Text),
            (9, 10, 9, 10, Token.Text),
            (10, 14, 10, 14, Token.Keyword),
        ]

    def test_mask_single_token(self):
        # source: 'abcdefghij'
        tokens = [(Token.Text, 'abcdefghij')]
        out, mapping = mask([(3, 7)], tokens, replacement='***')
        # before: [('abcdefghij', Text)]
        # after:  [('abc', Text), ('***', Text), ('hij', Text)]
        assert out == [
            (Token.Text, 'abc'),
            (Token.Text, '***'),
            (Token.Text, 'hij'),
        ]
        assert mapping == [
            (0, 3, 0, 3, Token.Text),
            (3, 6, 3, 7, Token.Text),   # masked: out len 3, src len 4
            (6, 9, 7, 10, Token.Text),
        ]

    def test_mask_spans_two_tokens(self):
        # source: 'abcdefghij', split into tokens at offsets 3 and 7
        tokens = [(Token.Text, 'abc'), (Token.Keyword, 'defg'),
                  (Token.Name, 'hij')]
        out, mapping = mask([(3, 8)], tokens, replacement='##')
        # The secret range [3, 8) spans tokens 2 and 3.  Only the corresponding
        # text is masked; the outer parts keep their token types.
        # before: [('abc', Text), ('defg', Keyword), ('hij', Name)]
        # after:  [('abc', Text), ('##', Text), ('##', Text), ('ij', Name)]
        assert out == [
            (Token.Text, 'abc'),
            (Token.Text, '##'),   # masked 'defg' (src 3..7)
            (Token.Text, '##'),   # masked 'h'    (src 7..8)
            (Token.Name, 'ij'),   # plain  'ij'   (src 8..10)
        ]
        # Subsequent mappings still point to the original source offsets.
        assert mapping == [
            (0, 3, 0, 3, Token.Text),
            (3, 5, 3, 7, Token.Text),   # masked, out len 2, src len 4
            (5, 7, 7, 8, Token.Text),   # masked, out len 2, src len 1
            (7, 9, 8, 10, Token.Name),
        ]

    def test_mask_spans_three_tokens(self):
        # source: 'abcdefghijklmnop', split into tokens of length 5, 5, 6
        tokens = [(Token.Text, 'abcde'), (Token.Keyword, 'fghij'),
                  (Token.Name, 'klmnop')]
        out, mapping = mask([(3, 13)], tokens, replacement='XX')
        # before: [('abcde', Text), ('fghij', Keyword), ('klmnop', Name)]
        # after:  [('abc', Text), ('XX', Text), ('XX', Text), ('XX', Text), ('nop', Name)]
        assert out == [
            (Token.Text, 'abc'),
            (Token.Text, 'XX'),   # masked 'de'    (src 3..5)
            (Token.Text, 'XX'),   # masked 'fghij' (src 5..10)
            (Token.Text, 'XX'),   # masked 'klm'   (src 10..13)
            (Token.Name, 'nop'),  # plain  'nop'   (src 13..16)
        ]
        assert mapping == [
            (0, 3, 0, 3, Token.Text),
            (3, 5, 3, 5, Token.Text),
            (5, 7, 5, 10, Token.Text),
            (7, 9, 10, 13, Token.Text),
            (9, 12, 13, 16, Token.Name),
        ]

    def test_replacement_longer_than_masked_range(self):
        # source: 'abcdefghij'
        tokens = [(Token.Text, 'abcdefghij')]
        out, mapping = mask([(3, 5)], tokens, replacement='XXXXX')
        # before: [('abcdefghij', Text)]
        # after:  [('abc', Text), ('XXXXX', Text), ('fghij', Text)]
        assert out == [
            (Token.Text, 'abc'),
            (Token.Text, 'XXXXX'),
            (Token.Text, 'fghij'),
        ]
        assert mapping == [
            (0, 3, 0, 3, Token.Text),
            (3, 8, 3, 5, Token.Text),    # masked: out len 5, src len 2
            (8, 13, 5, 10, Token.Text),
        ]

    def test_replacement_shorter_than_masked_range(self):
        # source: 'abcdefghij'
        tokens = [(Token.Text, 'abcdefghij')]
        out, mapping = mask([(2, 8)], tokens, replacement='Z')
        # before: [('abcdefghij', Text)]
        # after:  [('ab', Text), ('Z', Text), ('ij', Text)]
        assert out == [
            (Token.Text, 'ab'),
            (Token.Text, 'Z'),
            (Token.Text, 'ij'),
        ]
        assert mapping == [
            (0, 2, 0, 2, Token.Text),
            (2, 3, 2, 8, Token.Text),    # masked: out len 1, src len 6
            (3, 5, 8, 10, Token.Text),
        ]

    def test_multiple_ranges(self):
        # source: 'abcdefghij'
        tokens = [(Token.Text, 'abcdefghij')]
        out, mapping = mask([(1, 3), (6, 8)], tokens, replacement='*')
        # before: [('abcdefghij', Text)]
        # after:  [('a', Text), ('*', Text), ('def', Text), ('*', Text), ('ij', Text)]
        assert out == [
            (Token.Text, 'a'),
            (Token.Text, '*'),
            (Token.Text, 'def'),
            (Token.Text, '*'),
            (Token.Text, 'ij'),
        ]
        assert mapping == [
            (0, 1, 0, 1, Token.Text),
            (1, 2, 1, 3, Token.Text),
            (2, 5, 3, 6, Token.Text),
            (5, 6, 6, 8, Token.Text),
            (6, 8, 8, 10, Token.Text),
        ]

    def test_range_at_start(self):
        # source: 'abcdefghij'
        tokens = [(Token.Text, 'abcdefghij')]
        out, mapping = mask([(0, 4)], tokens, replacement='*')
        # before: [('abcdefghij', Text)]
        # after:  [('*', Text), ('efghij', Text)]
        assert out == [
            (Token.Text, '*'),
            (Token.Text, 'efghij'),
        ]
        assert mapping == [
            (0, 1, 0, 4, Token.Text),
            (1, 7, 4, 10, Token.Text),
        ]

    def test_range_at_end(self):
        # source: 'abcdefghij'
        tokens = [(Token.Text, 'abcdefghij')]
        out, mapping = mask([(7, 10)], tokens, replacement='*')
        # before: [('abcdefghij', Text)]
        # after:  [('abcdefg', Text), ('*', Text)]
        assert out == [
            (Token.Text, 'abcdefg'),
            (Token.Text, '*'),
        ]
        assert mapping == [
            (0, 7, 0, 7, Token.Text),
            (7, 8, 7, 10, Token.Text),
        ]

    def test_range_covers_whole_source(self):
        # source: 'abcdefghij'
        tokens = [(Token.Text, 'abcdefghij')]
        out, mapping = mask([(0, 10)], tokens, replacement='***')
        # before: [('abcdefghij', Text)]
        # after:  [('***', Text)]
        assert out == [(Token.Text, '***')]
        assert mapping == [(0, 3, 0, 10, Token.Text)]

    def test_touching_ranges_are_allowed(self):
        # source: 'abcdefghij'
        tokens = [(Token.Text, 'abcdefghij')]
        # ranges [1, 4) and [4, 7) touch at offset 4; no overlap
        out, mapping = mask([(1, 4), (4, 7)], tokens, replacement='*')
        # before: [('abcdefghij', Text)]
        # after:  [('a', Text), ('*', Text), ('*', Text), ('hij', Text)]
        assert out == [
            (Token.Text, 'a'),
            (Token.Text, '*'),
            (Token.Text, '*'),
            (Token.Text, 'hij'),
        ]
        assert mapping == [
            (0, 1, 0, 1, Token.Text),
            (1, 2, 1, 4, Token.Text),
            (2, 3, 4, 7, Token.Text),
            (3, 6, 7, 10, Token.Text),
        ]

    def test_ranges_are_sorted_automatically(self):
        # source: 'abcdefghij'
        tokens = [(Token.Text, 'abcdefghij')]
        # pass ranges out of order; they are sorted before processing
        out, mapping = mask([(6, 8), (1, 3)], tokens, replacement='*')
        assert out == [
            (Token.Text, 'a'),
            (Token.Text, '*'),
            (Token.Text, 'def'),
            (Token.Text, '*'),
            (Token.Text, 'ij'),
        ]

    def test_custom_mask_ttype(self):
        from pygments.token import Token as T
        tokens = [(T.Text, 'abcdefghij')]
        out, mapping = mask([(3, 5)], tokens, replacement='*',
                            mask_ttype=T.Generic.Strong)
        assert out == [
            (T.Text, 'abc'),
            (T.Generic.Strong, '*'),
            (T.Text, 'fghij'),
        ]
        assert mapping[1][4] == T.Generic.Strong

    def test_empty_range_has_no_effect(self):
        # source: 'abcdefghij'
        tokens = [(Token.Text, 'abcdefghij')]
        # empty range [3, 3) masks nothing
        out, mapping = mask([(3, 3)], tokens)
        assert out == tokens
        assert mapping == [(0, 10, 0, 10, Token.Text)]

    def test_empty_range_among_real_ranges(self):
        # source: 'abcdefghij'
        tokens = [(Token.Text, 'abcdefghij')]
        out, mapping = mask([(3, 3), (1, 4)], tokens, replacement='*')
        # the empty range is ignored; only [1, 4) is masked
        assert out == [
            (Token.Text, 'a'),
            (Token.Text, '*'),
            (Token.Text, 'efghij'),
        ]
        assert mapping == [
            (0, 1, 0, 1, Token.Text),
            (1, 2, 1, 4, Token.Text),
            (2, 8, 4, 10, Token.Text),
        ]

    def test_overlapping_ranges_are_rejected(self):
        tokens = [(Token.Text, 'abcdefghij')]
        with pytest.raises(ValueError, match='overlapping'):
            mask([(1, 4), (3, 7)], tokens)

    def test_overlapping_ranges_rejected_before_any_output(self):
        tokens = [(Token.Text, 'abcdefghij')]
        masker = TokenStreamMasker(ranges=[(1, 4), (3, 7)])
        gen = masker.filter(None, tokens)
        # validation happens on first iteration, before any yield
        with pytest.raises(ValueError, match='overlapping'):
            next(gen)
        # mapping stays empty: no partial masking was delivered
        assert masker.mapping == []

    def test_out_of_bounds_end_rejected(self):
        tokens = [(Token.Text, 'abcdefghij')]
        with pytest.raises(ValueError, match='exceeds source length'):
            mask([(0, 15)], tokens)

    def test_negative_offset_rejected(self):
        tokens = [(Token.Text, 'abcdefghij')]
        with pytest.raises(ValueError, match='negative'):
            mask([(-1, 5)], tokens)

    def test_start_greater_than_end_rejected(self):
        tokens = [(Token.Text, 'abcdefghij')]
        with pytest.raises(ValueError, match='start > end'):
            mask([(7, 3)], tokens)

    def test_invalid_range_type_rejected(self):
        tokens = [(Token.Text, 'abcdefghij')]
        with pytest.raises(ValueError, match=r'expected a \(start, end\) pair'):
            mask([(1,)], tokens)
        with pytest.raises(ValueError, match=r'expected a \(start, end\) pair'):
            mask(['not a range'], tokens)


    def test_surrogate_pair_splitting_rejected(self):
        # source with a surrogate pair: 'a😀b' (len 4)
        pair = chr(0xD83D) + chr(0xDE00)
        tokens = [(Token.Text, 'a' + pair + 'b')]
        # range [2, 3) splits the pair: high surrogate at 1, low at 2
        with pytest.raises(ValueError, match='surrogate'):
            mask([(2, 3)], tokens)

    def test_surrogate_pair_covered_whole_is_accepted(self):
        # source: 'a😀b' (len 4); range [1, 3) covers the whole pair
        pair = chr(0xD83D) + chr(0xDE00)
        tokens = [(Token.Text, 'a' + pair + 'b')]
        out, mapping = mask([(1, 3)], tokens, replacement='*')
        assert out == [
            (Token.Text, 'a'),
            (Token.Text, '*'),
            (Token.Text, 'b'),
        ]
        assert mapping == [
            (0, 1, 0, 1, Token.Text),
            (1, 2, 1, 3, Token.Text),
            (2, 3, 3, 4, Token.Text),
        ]

    def test_astral_char_not_split(self):
        # source with an astral char: 'a😀b' (len 3 in code points)
        tokens = [(Token.Text, 'a😀b')]
        out, mapping = mask([(1, 3)], tokens, replacement='*')
        # before: [('a😀b', Text)]
        # after:  [('a', Text), ('*', Text)]
        assert out == [
            (Token.Text, 'a'),
            (Token.Text, '*'),
        ]
        assert mapping == [
            (0, 1, 0, 1, Token.Text),
            (1, 2, 1, 3, Token.Text),
        ]

    def test_original_stream_values_not_modified(self):
        tokens = [(Token.Text, 'abc'), (Token.Keyword, 'defg'),
                  (Token.Name, 'hij')]
        original = [(t, v) for t, v in tokens]
        mask([(1, 3)], tokens)
        assert tokens == original

    def test_with_real_lexer(self):
        from pygments.lexers import PythonLexer
        lx = PythonLexer(ensurenl=False)
        lx.add_filter('mask', ranges=[(0, 3)], replacement='***')
        tokens = list(lx.get_tokens('def foo(): pass'))
        # 'def' (src 0..3) is masked; the rest is unchanged
        assert tokens[0] == (Token.Text, '***')
        assert ''.join(v for _, v in tokens[1:]) == ' foo(): pass'

    def test_with_real_lexer_spanning_tokens(self):
        from pygments.lexers import PythonLexer
        lx = PythonLexer(ensurenl=False)
        # 'secret = "hunter2"' ; mask the secret value inside the string
        text = 'secret = "hunter2"'
        # find the source offset of 'hunter2'
        start = text.index('hunter2')
        end = start + len('hunter2')
        lx.add_filter('mask', ranges=[(start, end)], replacement='******')
        tokens = list(lx.get_tokens(text))
        joined = ''.join(v for _, v in tokens)
        assert joined == 'secret = "******"'
        # the replacement is a single token
        assert any(v == '******' for _, v in tokens)

    def test_mapping_covers_output_contiguously(self):
        tokens = [(Token.Text, 'abcdefghij')]
        out, mapping = mask([(1, 3), (6, 8)], tokens, replacement='*')
        # output ranges are contiguous and start at 0
        assert mapping[0][0] == 0
        for i in range(1, len(mapping)):
            assert mapping[i][0] == mapping[i - 1][1]
        # the last output offset equals the total output length
        total_out = mapping[-1][1]
        expected = sum(len(v) for _, v in out)
        assert total_out == expected

    def test_mapping_src_ranges_cover_source(self):
        tokens = [(Token.Text, 'abcdefghij')]
        _, mapping = mask([(1, 3), (6, 8)], tokens, replacement='*')
        # plain segments' src ranges concatenate to the original source with
        # masked ranges removed; masked segments' src ranges are the secrets
        src_concat = ''.join(
            'abcdefghij'[s:e] for _, _, s, e, _ in mapping)
        assert src_concat == 'abcdefghij'
