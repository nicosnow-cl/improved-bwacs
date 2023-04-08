LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'


def same_line_print(content_list):
    total_lines = 0
    for content in content_list:
        for line in content:
            total_lines += 1
            print(line)

    clear_lines(total_lines)


def clear_lines(n=1):
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR)
