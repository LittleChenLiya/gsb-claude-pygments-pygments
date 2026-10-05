"""
    pygments.errors
    ~~~~~~~~~~~~~~~

    Error tracking for lexers.

    This module provides an explicit error list for lexers.  An
    :class:`ErrorRecord` is produced from an actual :token:`Token.Error`
    token emitted by the lexer — the token stream is not scanned a second
    time.  Each record captures the source line and column, the lexer
    state in which the error occurred, and the absolute byte (character)
    range of the error.

    If the lexer itself raises an exception (as opposed to emitting an
    Error token), the exception is recorded separately by
    :class:`LexerErrors`, together with the stage at which it occurred.

    :copyright: Copyright 2006-present by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

from bisect import bisect_right

from pygments.token import Error

__all__ = ['ErrorRecord', 'LexerErrors']


class ErrorRecord:
    """
    A record of a single lexer error.

    An error record is produced from an actual :token:`Token.Error` token
    emitted by the lexer.  It captures the location of the error in the
    source text and the lexer state in which it occurred.

    The state is only recorded if it is publicly nameable, i.e. it is a
    string that does not start with an underscore.  States generated for
    combined states (such as ``_tmp_0``) are not publicly nameable and are
    recorded as ``None``.

    Attributes:
        line: 1-based line number in the source text.
        column: 1-based column number (in characters, within the line).
        state: name of the lexer state active when the error was emitted,
            or ``None`` if the state is not publicly nameable.
        pos: absolute position of the start of the error in the source text.
        end: absolute position of the end of the error (exclusive).
    """
    __slots__ = ('column', 'end', 'line', 'pos', 'state')

    def __init__(self, line, column, state, pos, end):
        self.line = line
        self.column = column
        self.state = state
        self.pos = pos
        self.end = end

    def __repr__(self):
        return (f'ErrorRecord(line={self.line!r}, column={self.column!r}, '
                f'state={self.state!r}, pos={self.pos!r}, end={self.end!r})')

    def __eq__(self, other):
        if not isinstance(other, ErrorRecord):
            return NotImplemented
        return (self.line == other.line and
                self.column == other.column and
                self.state == other.state and
                self.pos == other.pos and
                self.end == other.end)

    def __hash__(self):
        return hash((self.line, self.column, self.state, self.pos, self.end))

    def asdict(self):
        """Return the record as a dictionary."""
        return {'line': self.line, 'column': self.column,
                'state': self.state, 'pos': self.pos, 'end': self.end}


class LexerErrors:
    """
    Collects error records from a lexer run.

    Records are produced from the actual :token:`Token.Error` tokens the
    lexer emits — the token stream is not scanned a second time.  The
    collector only inspects the token type and position; the token text
    is never stored.

    If the lexer itself raises an exception (as opposed to emitting an
    Error token), the exception is recorded separately in
    :attr:`exception` together with the :attr:`exception_stage` at which
    it occurred.  This allows lexer exceptions and Error tokens to be
    attributed independently.

    Attributes:
        records: list of :class:`ErrorRecord` objects.
        merge_adjacent: if ``True``, adjacent error records are merged into
            a single record spanning the whole range.  Defaults to
            ``False``.
        exception: the exception raised by the lexer, or ``None``.
        exception_stage: the stage at which the exception occurred (e.g.
            ``'lexing'``), or ``None``.
    """

    def __init__(self, merge_adjacent=False):
        self.merge_adjacent = merge_adjacent
        self.records = []
        self.exception = None
        self.exception_stage = None
        self._text = None
        self._line_starts = None

    def bind(self, text):
        """
        Bind the collector to the source ``text``.

        This is used to compute line and column numbers from absolute
        positions.  Returns ``self``.
        """
        self._text = text
        self._line_starts = None
        return self

    def _ensure_line_starts(self):
        if self._line_starts is None:
            starts = [0]
            for i, ch in enumerate(self._text):
                if ch == '\n':
                    starts.append(i + 1)
            self._line_starts = starts
        return self._line_starts

    def line_col(self, pos):
        """
        Return the ``(line, column)`` pair for an absolute position.
        """
        starts = self._ensure_line_starts()
        idx = bisect_right(starts, pos) - 1
        return idx + 1, pos - starts[idx] + 1

    def observe(self, pos, ttype, state=None):
        """
        Observe a token at position ``pos`` with type ``ttype``.

        If the token is an :token:`Token.Error` token, an error record is
        produced.  Only the type and position are inspected; the token
        text is not stored.

        Adjacent errors are merged into a single record only if
        :attr:`merge_adjacent` is ``True`` and the new error starts exactly
        at the end of the previous record.
        """
        if ttype is not Error:
            return
        line, column = self.line_col(pos)
        end = pos + 1
        if (self.merge_adjacent and self.records and
                self.records[-1].end == pos):
            prev = self.records[-1]
            self.records[-1] = ErrorRecord(
                prev.line, prev.column, prev.state, prev.pos, end)
        else:
            self.records.append(ErrorRecord(line, column, state, pos, end))

    def record_exception(self, exc, stage):
        """
        Record an exception raised by the lexer.

        ``stage`` is a string naming the stage at which the exception
        occurred (e.g. ``'lexing'``).  The exception is stored separately
        from the Error token records.
        """
        self.exception = exc
        self.exception_stage = stage

    def clear(self):
        """Remove all records and reset the exception state."""
        self.records = []
        self.exception = None
        self.exception_stage = None
        self._text = None
        self._line_starts = None

    def __bool__(self):
        return bool(self.records) or self.exception is not None

    def __len__(self):
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def __repr__(self):
        return (f'LexerErrors(records={self.records!r}, '
                f'exception={self.exception!r}, '
                f'exception_stage={self.exception_stage!r})')
