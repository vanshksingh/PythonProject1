import random
from typing import Literal
from langchain.tools import tool


FIRST_NAMES = [
    "Arjun", "Rohan", "Vivaan", "Ayaan", "Kabir",
    "Ishaan", "Dev", "Reyansh", "Aditya", "Vihaan",
]

LAST_NAMES = [
    "Sharma", "Mehta", "Verma", "Kapoor", "Bhatia",
    "Patel", "Singh", "Agarwal", "Malhotra", "Gupta",
]


def generate_random_name(style: str = "full") -> str:
    """
    Generate a random name.

    style options:
        - first
        - last
        - full
    """

    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)

    if style == "first":
        return first
    elif style == "last":
        return last
    else:
        return f"{first} {last}"


@tool
def random_name(style: Literal["first", "last", "full"] = "full") -> str:
    """
    Generate a random human name.

    Args:
        style: type of name to generate
            - first → first name only
            - last → last name only
            - full → full name

    Returns:
        Randomly generated name
    """

    return generate_random_name(style)