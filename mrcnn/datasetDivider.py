"""
Skinet (Segmentation of the Kidney through a Neural nETwork) Project

Copyright (c) 2021 Skinet Team
Licensed under the MIT License (see LICENSE for details)
Written by Adrien JAUGEY
"""
import math
from typing import List, Tuple

import numpy as np
import cv2

CV2_IMWRITE_PARAM = [int(cv2.IMWRITE_JPEG_QUALITY), 100, cv2.IMWRITE_PNG_COMPRESSION, 9]


def computeStartsOfInterval(maxVal: int, intervalLength=1024, min_overlap_part=0.33):
    """
    Divide the [0; maxVal] interval into a uniform distribution with at least min_overlap_part of overlapping
    :param maxVal: end of the base interval
    :param intervalLength: length of the new intervals
    :param min_overlap_part: min overlapping part of intervals, if less, adds intervals with length / 2 offset
    :return: list of starting coordinates for the new intervals
    """
    if maxVal <= intervalLength:
        return [0]
    nbDiv = math.ceil(maxVal / intervalLength)
    # Computing gap to get something that tends to a uniform distribution
    gap = (nbDiv * intervalLength - maxVal) / (nbDiv - 1)
    coordinates = []
    for i in range(nbDiv):
        coordinate = round(i * (intervalLength - gap))
        if i == nbDiv - 1:
            # Should not be useful but acting as a security
            coordinates.append(maxVal - intervalLength)
        else:
            coordinates.append(coordinate)
            # If gap is not enough, we add division with a intervalLength / 2 offset
            if gap < intervalLength * min_overlap_part:
                coordinates.append(coordinate + intervalLength // 2)
    return coordinates


def getMaxSizeForDivAmount(divAmount: int, intervalLength=1024, min_overlap_part=0.33):
    res = intervalLength * int(2. - min_overlap_part)
    while len(computeStartsOfInterval(res + 1, intervalLength, min_overlap_part)) <= divAmount:
        res += 1
    return res


def getDivisionsCount(xStarts: List[int], yStarts: List[int]):
    """
    Return the number of division for given starting x and y coordinates
    :param xStarts: the x-axis starting coordinates
    :param yStarts: the y-axis starting coordinates
    :return: number of divisions
    """
    return len(xStarts) * len(yStarts)


def getDivisionByID(xStarts: List[int], yStarts: List[int], idDivision: int,
                    divisionSize: [int, List[int], Tuple[int]] = 1024):
    """
    Return x and y starting and ending coordinates for a specific division
    :param xStarts: the x-axis starting coordinates
    :param yStarts: the y-axis starting coordinates
    :param idDivision: the ID of the division you want the coordinates. 0 <= ID < number of divisions
    :param divisionSize: length of the new intervals
    :return: x, xEnd, y, yEnd coordinates
    """
    # assert divisionSize is int or divisionSize is tuple
    if not 0 <= idDivision < len(xStarts) * len(yStarts):
        return -1, -1, -1, -1
    yIndex = idDivision // len(xStarts)
    xIndex = idDivision - yIndex * len(xStarts)

    x = xStarts[xIndex]
    xEnd = x + (divisionSize if type(divisionSize) is int else divisionSize[0])

    y = yStarts[yIndex]
    yEnd = y + (divisionSize if type(divisionSize) is int else divisionSize[1])
    return x, xEnd, y, yEnd


def getImageDivision(img, xStarts: List[int], yStarts: List[int], idDivision: int,
                     divisionSize: [int, List[int], Tuple[int]] = 1024):
    """
    Return the wanted division of an Image
    :param img: the base image
    :param xStarts: the x-axis starting coordinates
    :param yStarts: the y-axis starting coordinates
    :param idDivision: the ID of the division you want to get. 0 <= ID < number of divisions
    :param divisionSize: length of division side
    :return: the image division
    """
    # assert divisionSize is int or divisionSize is tuple
    x, xEnd, y, yEnd = getDivisionByID(xStarts, yStarts, idDivision, divisionSize)
    if len(img.shape) == 2:
        return img[y:yEnd, x:xEnd]
    else:
        return img[y:yEnd, x:xEnd, :]


def getBWCount(mask: np.ndarray):
    """
    Return number of black (0) and white (>0) pixels in a mask image
    :param mask: the mask image
    :return: number of black pixels, number of white pixels
    """
    mask = mask.astype(np.bool).flatten()
    totalPx = int(mask.shape[0])
    whitePx = int(np.sum(mask))
    return totalPx - whitePx, whitePx
