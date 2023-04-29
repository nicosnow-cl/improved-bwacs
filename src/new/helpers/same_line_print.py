from typing import List

LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'


def clear_lines(n=1):
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR)


def same_line_print(content_list: List[str], clear: bool = True):
    total_lines = 0
    for line in content_list:
        total_lines += 1
        print(line)

    if clear:
        clear_lines(total_lines)
