"""
    Pygments regex lexer tests
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    :copyright: Copyright 2006-present by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

import pytest

from pygments.token import Text, Whitespace, String, Comment
from pygments.lexer import (
    RegexLexer, default, RegexLexerCheckpoint,
    LexerTypeMismatchError, LexerConfigMismatchError, LexerStateError,
    ExtendedRegexLexer,
)


@pytest.fixture(scope='module')
def lexer():
    yield MyLexer()


class MyLexer(RegexLexer):
    """Test tuple state transitions including #pop."""
    tokens = {
        'root': [
            ('a', Text.Root, 'rag'),
            ('e', Text.Root),
            ('#', Text.Root, '#pop'),
            ('@', Text.Root, ('#pop', '#pop')),
            default(('beer', 'beer'))
        ],
        'beer': [
            ('d', Text.Beer, ('#pop', '#pop')),
        ],
        'rag': [
            ('b', Text.Rag, '#push'),
            ('c', Text.Rag, ('#pop', 'beer')),
        ],
    }


def test_tuple(lexer):
    toks = list(lexer.get_tokens_unprocessed('abcde'))
    assert toks == [
        (0, Text.Root, 'a'), (1, Text.Rag, 'b'), (2, Text.Rag, 'c'),
        (3, Text.Beer, 'd'), (4, Text.Root, 'e')]


def test_multiline(lexer):
    toks = list(lexer.get_tokens_unprocessed('a\ne'))
    assert toks == [
        (0, Text.Root, 'a'), (1, Whitespace, '\n'), (2, Text.Root, 'e')]


def test_default(lexer):
    toks = list(lexer.get_tokens_unprocessed('d'))
    assert toks == [(0, Text.Beer, 'd')]


def test_pop_empty_regular(lexer):
    toks = list(lexer.get_tokens_unprocessed('#e'))
    assert toks == [(0, Text.Root, '#'), (1, Text.Root, 'e')]


def test_pop_empty_tuple(lexer):
    toks = list(lexer.get_tokens_unprocessed('@e'))
    assert toks == [(0, Text.Root, '@'), (1, Text.Root, 'e')]


class SegmentedLexer(RegexLexer):
    """Lexer for testing segmented lexing with cross-line strings/comments."""
    tokens = {
        'root': [
            ('a', Text.Root),
            ('"', String, 'string'),
            ('#', Comment, 'comment'),
        ],
        'string': [
            ('"', String, '#pop'),
            ('[^"]+', String),
        ],
        'comment': [
            ('\n', Comment, '#pop'),
            ('[^\n]+', Comment),
        ],
    }


@pytest.fixture(scope='module')
def seg_lexer():
    yield SegmentedLexer()


def _segment(lexer, lines):
    """Lex a list of lines segment by segment, returning all tokens."""
    tokens = []
    checkpoint = None
    for line in lines:
        toks, checkpoint = lexer.lex_segment(line, checkpoint)
        tokens.extend(toks)
    return tokens


def test_segmented_cross_line_string(seg_lexer):
    """Cross-line strings: token types and offsets must equal one-shot."""
    lines = ['a"hello\n', 'world\n', 'end"\n']
    full = ''.join(lines)
    one_shot = list(seg_lexer.get_tokens_unprocessed(full))
    segmented = _segment(seg_lexer, lines)
    assert segmented == one_shot


def test_segmented_cross_line_comment(seg_lexer):
    """Cross-line comments: token types and offsets must equal one-shot."""
    lines = ['a# comment\n', 'still comment\n', 'done\n']
    full = ''.join(lines)
    one_shot = list(seg_lexer.get_tokens_unprocessed(full))
    segmented = _segment(seg_lexer, lines)
    assert segmented == one_shot


def test_segmented_checkpoint_recreation(seg_lexer):
    """Save checkpoint midway, recreate lexer, continue consistently."""
    lines = ['a"hello\n', 'world\n', 'end"\n']
    full = ''.join(lines)
    one_shot = list(seg_lexer.get_tokens_unprocessed(full))

    # Lex the first line, then recreate the lexer from the checkpoint.
    toks1, cp = seg_lexer.lex_segment(lines[0])
    new_lexer = SegmentedLexer()
    toks2, cp = new_lexer.lex_segment(lines[1], cp)
    toks3, cp = new_lexer.lex_segment(lines[2], cp)
    segmented = toks1 + toks2 + toks3
    assert segmented == one_shot


def test_segmented_absolute_offsets(seg_lexer):
    """Absolute token offsets must not reset across segments."""
    lines = ['a"hello\n', 'world\n', 'end"\n']
    tokens = _segment(seg_lexer, lines)
    # Verify that offsets are monotonically increasing and match the
    # cumulative length of the source.
    full = ''.join(lines)
    for i, (pos, tok, val) in enumerate(tokens):
        assert pos == sum(len(v) for _, _, v in tokens[:i])
        assert full[pos:pos + len(val)] == val


def test_segmented_config_mismatch():
    """Checkpoint with different config must raise ConfigMismatch."""
    lexer1 = SegmentedLexer(foo='bar')
    lexer2 = SegmentedLexer(foo='baz')
    _, cp = lexer1.lex_segment('a"hello')
    with pytest.raises(LexerConfigMismatchError):
        lexer2.lex_segment('world"', cp)


def test_segmented_type_mismatch(seg_lexer):
    """Checkpoint with different lexer type must raise TypeMismatch."""
    class OtherLexer(RegexLexer):
        tokens = {'root': [('b', Text.Root)]}
    other = OtherLexer()
    _, cp = seg_lexer.lex_segment('a"hello')
    with pytest.raises(LexerTypeMismatchError):
        other.lex_segment('world"', cp)


def test_segmented_override_rejected():
    """Lexers overriding get_tokens_unprocessed must be rejected."""
    class CustomLexer(RegexLexer):
        tokens = {'root': [('a', Text.Root)]}
        def get_tokens_unprocessed(self, text, stack=('root',)):
            yield 0, Text.Root, text
    custom = CustomLexer()
    with pytest.raises(LexerStateError):
        custom.lex_segment('a')


def test_segmented_extended_rejected():
    """ExtendedRegexLexer must be rejected."""
    class MyExtendedLexer(ExtendedRegexLexer):
        tokens = {'root': [('a', Text.Root)]}
    ext = MyExtendedLexer()
    with pytest.raises(LexerStateError):
        ext.lex_segment('a')


def test_segmented_invalid_state(seg_lexer):
    """Checkpoint with unknown state must raise StateError."""
    fake_cp = RegexLexerCheckpoint(
        statestack=('nonexistent',),
        pos=0,
        pending=None,
        lexer_id=seg_lexer._lexer_id(),
        config_hash=seg_lexer._config_hash(),
    )
    with pytest.raises(LexerStateError):
        seg_lexer.lex_segment('a', fake_cp)


def test_segmented_root_state(seg_lexer):
    """Lexing in root state should produce no pending text."""
    tokens, cp = seg_lexer.lex_segment('abc')
    assert cp.statestack == ('root',)
    assert cp.pending is None
    assert cp.pos == 3


def test_segmented_complete_string(seg_lexer):
    """A complete string in one segment should produce no pending text."""
    tokens, cp = seg_lexer.lex_segment('a"hello"')
    assert cp.statestack == ('root',)
    assert cp.pending is None
    assert cp.pos == 8
    assert tokens == [
        (0, Text.Root, 'a'),
        (1, String, '"'),
        (2, String, 'hello'),
        (7, String, '"'),
    ]


class NestedLexer(RegexLexer):
    """Lexer for testing segmented lexing with nested states."""
    tokens = {
        'root': [
            ('a', Text.Root),
            ('"', String, 'string'),
        ],
        'string': [
            ('"', String, '#pop'),
            ('#{', String.Interpol, 'interpol'),
            ('[^"#]+', String),
        ],
        'interpol': [
            ('\n', String.Interpol),
            ('}', String.Interpol, '#pop'),
            ('[^}\n]+', String.Interpol),
        ],
    }


@pytest.fixture(scope='module')
def nested_lexer():
    yield NestedLexer()


def test_segmented_nested_states(nested_lexer):
    """Nested states: tokens and offsets must equal one-shot."""
    lines = ['a"hello#{world\n', '}end', '"\n']
    full = ''.join(lines)
    one_shot = list(nested_lexer.get_tokens_unprocessed(full))
    segmented = _segment(nested_lexer, lines)
    assert segmented == one_shot


def test_segmented_nested_states_pending(nested_lexer):
    """Pending text must only include text consumed in the current state."""
    lines = ['a"hello#{world\n', '}end', '"\n']
    # After the second line, we should be in the 'string' state with
    # pending text 'end' (not 'world\n}end').
    _, cp = nested_lexer.lex_segment(lines[0])
    _, cp = nested_lexer.lex_segment(lines[1], cp)
    assert cp.statestack == ('root', 'string')
    assert cp.pending == 'end'
    assert cp.pos == 19
