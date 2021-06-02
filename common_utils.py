"""
Skinet (Segmentation of the Kidney through a Neural nETwork) Project
Common display/math methods

Copyright (c) 2021 Skinet Team
Licensed under the MIT License (see LICENSE for details)
Written by Adrien JAUGEY
"""
from datetime import datetime
import numpy as np


def progressBar(value, maxValue, prefix="", suffix="", forceNewLine=False, size=20, full='█', cursor='▒',
                empty='░'):
    """
    Prints a progress bar
    Based on https://stackoverflow.com/questions/6169217/replace-console-output-in-python
    :param value: the current progress value
    :param maxValue: the maximum value that could be given
    :param prefix: text to display before the progress bar
    :param suffix: text to display after the progress bar
    :param forceNewLine: False by default, if True a new line will be created even if not at the end
    :param size: size of the bar itself
    :param full: the character to use for the completed part of the bar
    :param cursor: the character to use for the current position
    :param empty: the character to use for the empty part of the bar
    :return: None
    """
    percent = float(value) / maxValue
    nbFullChar = int(percent * size)
    bar = full * nbFullChar + (cursor if percent > 0 and nbFullChar < size else "")
    emptyBar = empty * (size - len(bar))
    print(f'\r{prefix} {bar}{emptyBar} {percent: 6.2%} {suffix}',
          end='\n' if value == maxValue or forceNewLine else "", flush=True)


def progressText(value, maxValue, onlyRaw=False, onlyPercent=False):
    if onlyRaw and onlyPercent:
        return None
    elif onlyRaw:
        return f"({value}/{maxValue})"
    else:
        percent = float(value) / maxValue
        if onlyPercent:
            return f"({percent:06.2%})"
        else:
            return f"({value}/{maxValue} | {percent:06.2%})"


def formatTime(seconds: int, minutes: int = 0, hours: int = 0):
    """
    Returns a string representing the given time
    :param seconds: number of seconds
    :param minutes: if given, number of minutes adding to seconds
    :param hours:  if given, number of hours adding to seconds
    :return: formated time as string of "hh:mm:ss" format
    """
    seconds += minutes * 60 + hours * 3600
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    hText = f"{h:02d}:" if h != 0 else ""
    mText = f"{m:02d}:" if h + m != 0 else ""
    sText = f"{s:02d}" if h + m != 0 else f"{s:02d}s"
    return f"{hText}{mText}{sText}"


def formatDate(date: datetime = None, dateOnly=False, timeOnly=False):
    """
    Returns date as string
    :param date: datetime.datetime object to use specific date, current date and time by default
    :param dateOnly: if True, will only display the date, not the time
    :param timeOnly: if True, will only display the time, not the date
    :return: formatted date as string
    """
    if date is None:
        date = datetime.now()
    else:
        assert type(date) is datetime, "Provide date as datetime.datetime object"
    dateFormat = '%Y-%m-%d'
    timeFormat = '%H-%M-%S'
    if dateOnly and timeOnly:
        outputFormat = ""
    elif dateOnly:
        outputFormat = dateFormat
    elif timeOnly:
        outputFormat = timeFormat
    else:
        outputFormat = f"{dateFormat}_{timeFormat}"
    return date.strftime(outputFormat)


def format_number(num, maxLength=None):
    """
    Formats a number using a metric prefix such as K (Kilo), M (Mega)... up to Y (Yotta)
    :param num: the number to format
    :param maxLength: maximum length of the formatted number, will reduce number of decimals from 2 to 0 depending on
                      the length but will at least return abs(num / 1000^N) followed by the metric prefix for 1000^N.
    :return: the formatted number with up to 2 decimal digits
    """
    # https://stackoverflow.com/questions/579310/formatting-long-numbers-as-strings-in-python
    suffixes = ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']
    magnitude = 0
    isInt = np.issubdtype(type(num), np.integer)
    while abs(num) >= 1000 and magnitude < len(suffixes) - 1:
        magnitude += 1
        num /= 1000.0

    if magnitude == 0 and isInt:
        return str(int(num))

    if maxLength is not None:
        possibleText = [f"{num:.2f}{suffixes[magnitude]}",
                        f"{num:.1f}{suffixes[magnitude]}",
                        f"{num:.0f}{suffixes[magnitude]}"]
        for text in possibleText:
            if len(text) <= maxLength:
                return text
        return possibleText[-1]
    return f"{num:.1f}{suffixes[magnitude]}"


def combination(setSize, combinationSize):
    """
    Computes the number of k-combinations in a set
    Source : https://python.jpvweb.com/python/mesrecettespython/doku.php?id=combinaisons
    :param setSize: number of elements in the set
    :param combinationSize: number of elements in a combination
    :return: number of k-combinations
    """
    if combinationSize > setSize // 2:
        combinationSize = setSize - combinationSize
    x = 1
    y = 1
    i = setSize - combinationSize + 1
    while i <= setSize:
        x = (x * i) // y
        y += 1
        i += 1
    return x
