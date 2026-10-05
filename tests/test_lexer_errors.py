"""
    Explicit lexer error list tests
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Tests for the error-reporting facility in :mod:`pygments.lexer`.

    :copyright: Copyright 2006-2026 by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

import pytest

from pygments.lexer import (RegexLexer, ExtendedRegexLexer,
                            ErrorCollector, IncrementalLexer, using,
                            DelegatingLexer)
from pygments.token import Text, Name, Other, Error, Whitespace


class PlainLexer(RegexLexer):
    """A minimal lexer with a named pushed state."""
    tokens = {
        'root': [
            (r"'", Text, 'string'),
            (r'[A-Za-z_]+', Name),
            (r'\s+', Whitespace),
        ],
        'string': [
            (r"[^']+", Text),
            (r"'", Text, '#pop'),
        ],
    }


class ExtendedLexer(ExtendedRegexLexer):
    tokens = {
        'root': [
            (r"'", Text, 'string'),
            (r'[A-Za-z_]+', Name),
            (r'\s+', Whitespace),
        ],
        'string': [
            (r"[^']+", Text),
            (r"'", Text, '#pop'),
        ],
    }


class RaisingActionLexer(RegexLexer):
    """A rule callback raises instead of emitting a token."""
    tokens = {'root': [
        (r'x', Text),
        (r'!', lambda lexer, match: (_ for _ in ()).throw(
            RuntimeError('action exploded'))),
        (r'@', Error),
        (r'.', Text),
    ]}


class RaisingMatchLexer(RegexLexer):
    """A rule matcher raises during the matching phase."""
    tokens = {'root': [(r'x', Text)]}

    def __init__(self, **options):
        RegexLexer.__init__(self, **options)
        self._tokens['root'][0] = (
            lambda *a, **k: 1 / 0, Text, None)


BAD = "ok $$ x\nab $ y\n"
# Split at token boundaries: after 'ok $$', after the following newline.
SEGMENTS = [BAD[:5], BAD[5:8], BAD[8:]]


def _oneshot_errors(lexer, source):
    collector = ErrorCollector(source)
    token = collector.activate()
    try:
        list(lexer.get_tokens_unprocessed(source))
    finally:
        collector.restore(token)
    return collector.errors


# --------------------------------------------------------------------------
# One-shot recording
#

def test_oneshot_records_error_positions():
    errors = _oneshot_errors(PlainLexer(), BAD)
    assert [(e.line, e.column) for e in errors] == [
        (1, 3), (1, 4), (2, 3)]
    assert [(e.start, e.end) for e in errors] == [
        (3, 4), (4, 5), (11, 12)]
    assert all(e.origin == 'token' for e in errors)
    assert all(e.token_type is Error for e in errors)
    assert all(e.state == ('root',) for e in errors)


def test_error_records_match_actual_error_tokens():
    lexer = PlainLexer()
    stream = list(lexer.get_tokens_unprocessed(BAD))
    stream_positions = sorted(i for i, t, v in stream if t is Error)
    recorded = sorted(e.start for e in _oneshot_errors(lexer, BAD))
    assert stream_positions == recorded


def test_get_tokens_populates_errors_attribute():
    lexer = PlainLexer()
    list(lexer.get_tokens(BAD))
    assert len(lexer.errors) == 3
    assert (lexer.errors[0].line, lexer.errors[0].column) == (1, 3)
    list(lexer.get_tokens('clean text\n'))
    assert lexer.errors == []


def test_error_tokens_remain_and_order_unchanged():
    clean = PlainLexer()
    instrumented = PlainLexer()
    plain = [(t, v) for _, t, v in clean.get_tokens_unprocessed(BAD)]
    streamed = list(instrumented.get_tokens(BAD))
    assert streamed == plain
    assert any(t is Error for t, v in streamed)


def test_state_recorded_in_pushed_state():
    # A character the 'string' state does not know about yields an Error
    # while that state is active.
    class StringStrict(RegexLexer):
        tokens = {
            'root': [(r"'", Text, 'string'), (r'.', Text)],
            'string': [(r"[a-z ]+", Text), (r"'", Text, '#pop')],
        }
    errors = _oneshot_errors(StringStrict(), "a 'x $ b")
    assert errors and errors[0].state == ('root', 'string')
    assert (errors[0].line, errors[0].column) == (1, 5)


def test_state_only_contains_public_names():
    # combined() is compiled into an internal '_tmp_...' state; that name
    # must never leak into the recorded state.
    from pygments.lexer import combined
    from pygments.lexer import _state_sink, LexerContext

    class WithCombinedState(RegexLexer):
        tokens = {
            'root': [(r'a', Text, combined('public'))],
            'public': [(r'.', Error)],
        }

    # Observe the raw stack through the internal sink: it contains the
    # internal combined state name.
    sink = LexerContext('a!', 0)
    token = _state_sink.set(sink)
    try:
        list(WithCombinedState().get_tokens_unprocessed('a!'))
    finally:
        _state_sink.reset(token)
    assert any(name.startswith('_') for name in sink.stack)

    # The recorded (public) state hides it.
    errors = _oneshot_errors(WithCombinedState(), 'a!')
    assert errors
    assert all(not name.startswith('_') for name in errors[0].state)
    assert errors[0].state == ('root',)


# --------------------------------------------------------------------------
# Incremental / segmented lexing
#

@pytest.mark.parametrize('lexer_cls', [PlainLexer, ExtendedLexer])
def test_incremental_matches_oneshot_positions(lexer_cls):
    collector = ErrorCollector(BAD)
    token = collector.activate()
    try:
        one = list(lexer_cls().get_tokens_unprocessed(BAD))
    finally:
        collector.restore(token)

    incremental = IncrementalLexer(lexer_cls())
    got = list(incremental.feed(SEGMENTS[0]))
    got += list(incremental.feed(SEGMENTS[1]))
    got += list(incremental.finish(SEGMENTS[2]))

    assert got == one
    assert incremental.errors == collector.errors


def test_incremental_state_is_continuous():
    incremental = IncrementalLexer(PlainLexer())
    list(incremental.feed("ab '"))
    assert incremental.state == ('root', 'string')
    list(incremental.finish("cd'\n"))
    assert incremental.state == ('root',)


def test_incremental_absolute_positions_across_lines():
    incremental = IncrementalLexer(PlainLexer())
    list(incremental.feed(SEGMENTS[0]))
    list(incremental.feed(SEGMENTS[1]))
    list(incremental.finish(SEGMENTS[2]))
    positions = [(e.start, e.end, e.line, e.column)
                 for e in incremental.errors]
    assert positions == [(3, 4, 1, 3), (4, 5, 1, 4), (11, 12, 2, 3)]


# --------------------------------------------------------------------------
# Callback and merge options
#

def test_callback_receives_type_and_position_only():
    received = []

    def callback(**kwargs):
        assert set(kwargs) == {
            'token_type', 'start', 'end', 'line', 'column', 'state'}
        received.append(kwargs)

    lexer = PlainLexer(error_callback=callback)
    list(lexer.get_tokens(BAD))
    assert len(received) == 3
    assert received[0]['token_type'] is Error
    assert (received[0]['start'], received[0]['end']) == (3, 4)
    assert (received[0]['line'], received[0]['column']) == (1, 3)


def test_adjacent_errors_not_merged_by_default():
    assert len(_oneshot_errors(PlainLexer(), "$$")) == 2


def test_adjacent_errors_merged_when_option_set():
    collector = ErrorCollector("$$", merge_adjacent_errors=True)
    token = collector.activate()
    try:
        list(PlainLexer().get_tokens_unprocessed("$$"))
    finally:
        collector.restore(token)
    assert len(collector.errors) == 1
    assert (collector.errors[0].start,
            collector.errors[0].end) == (0, 2)


def test_merge_across_segment_boundary():
    incremental = IncrementalLexer(PlainLexer(),
                                   merge_adjacent_errors=True)
    list(incremental.feed("$"))
    list(incremental.finish("$"))
    assert len(incremental.errors) == 1
    assert (incremental.errors[0].start,
            incremental.errors[0].end) == (0, 2)


def test_non_adjacent_errors_remain_separate_under_merge():
    collector = ErrorCollector("$a$", merge_adjacent_errors=True)
    token = collector.activate()
    try:
        list(PlainLexer().get_tokens_unprocessed("$a$"))
    finally:
        collector.restore(token)
    assert len(collector.errors) == 2


# --------------------------------------------------------------------------
# Exceptions vs. error tokens
#

def test_action_exception_is_separately_attributed():
    lexer = RaisingActionLexer()
    emitted = []
    with pytest.raises(RuntimeError):
        for tok in lexer.get_tokens("xx!yy"):
            emitted.append(tok)
    assert emitted == [(Text, 'x'), (Text, 'x')]
    assert lexer.failed_stage == 'action'
    exceptions = [e for e in lexer.errors if e.origin == 'exception']
    assert len(exceptions) == 1
    assert exceptions[0].stage == 'action'
    assert isinstance(exceptions[0].error, RuntimeError)
    assert (exceptions[0].line, exceptions[0].column) == (1, 2)


def test_match_exception_records_match_stage():
    lexer = RaisingMatchLexer()
    with pytest.raises(ZeroDivisionError):
        list(lexer.get_tokens("xxx"))
    assert lexer.failed_stage == 'match'
    assert lexer.errors and lexer.errors[0].origin == 'exception'
    assert lexer.errors[0].stage == 'match'


def test_error_token_and_exception_are_distinct_origins():
    lexer = RaisingActionLexer()
    with pytest.raises(RuntimeError):
        # '@' yields a genuine Error token, '!' then raises.
        list(lexer.get_tokens("@!"))
    origins = sorted(e.origin for e in lexer.errors)
    assert origins == ['exception', 'token']
    token_error = next(e for e in lexer.errors if e.origin == 'token')
    assert (token_error.start, token_error.end) == (0, 1)


# --------------------------------------------------------------------------
# Nested / delegating lexers
#

class _Inner(RegexLexer):
    tokens = {'root': [
        (r'[a-z]+', Name),
        (r'.', Error),
    ]}


class _Outer(RegexLexer):
    tokens = {
        'root': [
            (r'<', Text, 'inner'),
            (r'[A-Z]+', Name),
            (r'.', Text),
        ],
        'inner': [
            (r'[^>]+', using(_Inner)),
            (r'>', Text, '#pop'),
        ],
    }


def test_nested_using_errors_recorded_once_with_outer_position():
    lexer = _Outer()
    list(lexer.get_tokens("AB<ab@!>CD"))
    assert [(e.start, e.end) for e in lexer.errors] == [(5, 6), (6, 7)]


def test_delegating_lexer_records_errors_once():
    # The language lexer hands needle ('Other') runs to the root lexer; a
    # '#' inside such a run is an error produced by the root lexer.
    class Root(RegexLexer):
        tokens = {'root': [(r'[a-z]+', Name), (r'#', Error),
                           (r'.', Text)]}

    class Language(RegexLexer):
        tokens = {'root': [(r'\{[^}]*\}', Other), (r'#', Other),
                           (r'.', Text)]}

    class Del(DelegatingLexer):
        def __init__(self, **options):
            DelegatingLexer.__init__(self, Root, Language, **options)

    lexer = Del()
    stream = list(lexer.get_tokens("ab{xx}#cd"))
    assert any(t is Error for t, v in stream)
    records = [(e.start, e.end) for e in lexer.errors]
    # '#' is at original offset 6 and recorded a single time.
    assert records == [(6, 7)]
    assert len(records) == len(set(records))


def test_delegating_lexer_language_error_recorded_once():
    class Root(RegexLexer):
        tokens = {'root': [(r'[a-z]+', Name), (r'.', Text)]}

    class Language(RegexLexer):
        tokens = {'root': [(r'\{[^}]*\}', Other), (r'@', Error),
                           (r'.', Text)]}

    class Del(DelegatingLexer):
        def __init__(self, **options):
            DelegatingLexer.__init__(self, Root, Language, **options)

    lexer = Del()
    stream = list(lexer.get_tokens("ab@{xx}cd"))
    assert any(t is Error for t, v in stream)
    # '@' at offset 2 comes from the language lexer; recorded once.
    assert [(e.start, e.end) for e in lexer.errors] == [(2, 3)]
