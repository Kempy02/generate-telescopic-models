# io_modules/string_helpers.py

def first_letter_and_number(s: str) -> str:
    """
    Return the first alphabetic character (uppercased) concatenated with the first digit found in s.
    Examples:
        'BENDING1' -> 'B1'
        'linear2'  -> 'L2'
    Raises ValueError if no letter or no digit is present.
    """
    if not isinstance(s, str):
        raise TypeError("Input must be a string")
    letter = None
    digit = None
    for ch in s:
        if letter is None and ch.isalpha():
            letter = ch.upper()
        if digit is None and ch.isdigit():
            digit = ch
        if letter is not None and digit is not None:
            return f"{letter}{digit}"
    raise ValueError("String must contain at least one letter and one digit")