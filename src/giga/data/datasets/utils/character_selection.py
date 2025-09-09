from pathlib import Path


def get_slice_size(available_chars: list[str], flag: str) -> int | str:
    """Get number of characters based on selection flag."""
    match flag:
        case "all":
            return len(available_chars)
        case "half":
            return len(available_chars) // 2
        case "quarter":
            return len(available_chars) // 4
        case "eighth":
            return len(available_chars) // 8
        case _:
            return flag


def filter_characters(
    available_chars: list[str],
    selection: list[str | int] | str,
    exclusions: Path | None = None,
) -> list[str]:
    """Common character filtering logic."""
    # Handle exclusions
    if exclusions is not None:
        with open(exclusions, "r") as f:
            excluded_chars = set(f.read().splitlines())
        available_chars = [char for char in available_chars if char not in excluded_chars]

    # Handle different selection types
    if isinstance(selection, str):
        assert selection in available_chars, f"Character {selection} not found in available characters."
        return [selection]
    elif isinstance(selection[0], str):
        num_chars = get_slice_size(available_chars, selection[0])
        if isinstance(num_chars, int):
            return available_chars[:num_chars]
        return [num_chars]
    elif len(selection) == 1:
        return available_chars[: selection[0]]
    elif len(selection) == 2:
        start, end = selection
        return available_chars[start:end]
    elif len(selection) == 3:
        start, end, step = selection
        return available_chars[start:end:step]
    else:
        return [available_chars[i] for i in selection]


def select_characters_neuman(
    data_dir: Path,
    selection: list[str | int] | str,
    exclusions: Path | None = None,
) -> list[str]:
    """Select characters for Neuman dataset."""
    if isinstance(selection[0], str):
        assert len(selection) == 1, "Neuman dataset can only select one character at a time."
    available_chars = sorted([item.name for item in data_dir.iterdir() if item.is_dir()])

    if isinstance(selection[0], int):
        character = [available_chars[selection[0]]]
    else:
        assert selection[0] in available_chars, f"Character {selection[0]} not found in available characters."
        character = selection

    return character


def select_characters_mvh(
    data_dir: Path,
    selection: list[str | int] | str,
    exclusions: Path | None = None,
) -> list[str]:
    """Select characters for MVH or MVH++ dataset."""
    available_chars = sorted([item.name for item in data_dir.iterdir() if item.is_dir()])
    return filter_characters(available_chars, selection, exclusions)


def select_characters_dna(
    data_dir: Path,
    selection: list[str | int] | str,
    exclusions: Path | None = None,
) -> list[str]:
    """Select characters for DNA dataset."""
    main_dir = data_dir / "main"
    available_chars = sorted([item.stem for item in main_dir.glob("*.smc")])
    return filter_characters(available_chars, selection, exclusions)
