class AnsiColor:
    RESET = "\033[0m"

    COLORS = {
        "black": "30",
        "red": "91",
        "green": "92",
        "yellow": "93",
        "blue": "94",
        "magenta": "95",
        "cyan": "96",
        "white": "97",
    }

    @staticmethod
    def colorize(text, color="white", bold=False):
        color_code = AnsiColor.COLORS.get(color.lower(), "97")
        codes = []

        if bold:
            codes.append("1")
        codes.append(color_code)

        return f"\033[{';'.join(codes)}m{text}{AnsiColor.RESET}"
