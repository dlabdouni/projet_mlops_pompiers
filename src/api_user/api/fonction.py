# Fonction pour formater le temps en minutes (arrondi à la minute supérieure)
def format_time(minutes: int) -> str:
    if minutes == 1:
        minute_str = "1 min"
    else:
        minute_str = f"{minutes} mins"

    return minute_str

def format_real_time(minutes: int, seconds: int) -> str:
    minute_str = "1 min" if minutes == 1 else f"{minutes} mins"
    second_str = "1 sec" if seconds == 1 else f"{seconds} secs"

    if minutes == 0:
        return second_str
    elif seconds == 0:
        return minute_str
    else:
        return f"{minute_str} {second_str}"