import re


def remove_invalid_char(string: str):
    return re.sub(r'[\\/:*?"<>|]+', '', string)
