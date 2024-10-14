def custom_anonymizer(text, found_nes, replacement_strategy: str = "category"):
    offset = 0
    for ne in found_nes:
        if replacement_strategy == "category":
            replacement = f'[{ne["CATEGORY"]}]'
        else:
            replacement = "[REDACTED]"

        start_position = ne["START"] + offset
        end_position = ne["END"] + offset
        new_end_position = start_position + len(replacement)

        prefix = text[:start_position]
        suffix = text[end_position:]

        text = prefix + replacement + suffix

        offset += new_end_position - end_position

    return text
