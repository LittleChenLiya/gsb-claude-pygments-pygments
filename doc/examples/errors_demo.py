"""
    Lexer error tracking demo
    ~~~~~~~~~~~~~~~~~~~~~~~~~

    Demonstrates the explicit error list provided by `LexerErrors`.

    Run with::

        python doc/examples/errors_demo.py

    :copyright: Copyright 2006-present by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

from pygments.lexer import LexerErrors, lex_incremental
from pygments.lexers import PythonLexer
from pygments.token import Error


def main():
    lexer = PythonLexer()

    # A "bad string": source code that contains characters the Python
    # lexer cannot recognize (here, NUL bytes).
    bad_source = 'x = 1\ny = \x00 + 2\nz = \x00\n'

    print('=== One-time lex with error tracking ===')
    errors = LexerErrors()
    tokens = list(lexer.get_tokens(bad_source, errors=errors))

    print('Token stream (Error tokens are kept):')
    for ttype, value in tokens:
        marker = '  <-- Error' if ttype is Error else ''
        print(f'  {value!r:12} {ttype!s}{marker}')

    print()
    print('Error records (from actual Token.Error output):')
    for record in errors.records:
        print(f'  line={record.line} col={record.column} '
              f'state={record.state} pos={record.pos} end={record.end}')

    print()
    print('=== Incremental lex (state and absolute positions persist) ===')
    # Split the source at a newline boundary and lex each segment.
    full = lexer._preprocess_lexer_input(bad_source)
    mid = full.index('\n') + 1
    segments = [full[:mid], full[mid:]]

    inc_errors = LexerErrors()
    inc_tokens, _ = lex_incremental(lexer, segments, errors=inc_errors)

    print('Incremental error positions match one-time lex:')
    for record in inc_errors.records:
        print(f'  line={record.line} col={record.column} '
              f'state={record.state} pos={record.pos} end={record.end}')
    print(f'Match: {inc_errors.records == errors.records}')
    print(f'Token order unchanged: {inc_tokens == tokens}')

    print()
    print('=== Merge adjacent errors (optional) ===')
    merged = LexerErrors(merge_adjacent=True)
    bad = 'a\x00\x00b\n'
    list(lexer.get_tokens(bad, errors=merged))
    print(f'Without merge: {len(lexer.get_errors(bad).records)} records')
    print(f'With merge:    {len(merged.records)} records')
    for record in merged.records:
        print(f'  pos={record.pos} end={record.end}')


if __name__ == '__main__':
    main()
