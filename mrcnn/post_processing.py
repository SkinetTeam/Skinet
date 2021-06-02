"""
Skinet (Segmentation of the Kidney through a Neural nETwork) Project

Copyright (c) 2021 Skinet Team
Licensed under the MIT License (see LICENSE for details)
Written by Adrien JAUGEY
"""
from enum import Enum
from time import time
from typing import List

import cv2
import numpy as np

from common_utils import combination, formatTime, progressBar, progressText
from mrcnn import datasetDivider as dD
from mrcnn.config import Config
from mrcnn.utils import get_bbox_area, get_bboxes_intersection, global_bbox, shift_bbox
from mrcnn.utils import minimize_mask, expand_mask, expand_masks, get_mask_area


def fuse_results(results, image_shape, division_size=1024, min_overlap_part=0.33):
    """
    Fuse results of multiple predictions (divisions for example)
    :param results: list of the results of the predictions
    :param image_shape: the input image shape
    :param division_size: Size of a division
    :param min_overlap_part: Minimum overlap of divisions
    :return: same structure contained in results
    """
    # Getting base input image information
    div_side_length = results[0]['masks'].shape[0]
    use_mini_mask = division_size != "noDiv" and division_size != div_side_length
    height, width, _ = image_shape
    xStarts = [0] if division_size == "noDiv" else dD.computeStartsOfInterval(width, division_size,
                                                                              min_overlap_part=min_overlap_part)
    yStarts = [0] if division_size == "noDiv" else dD.computeStartsOfInterval(height, division_size,
                                                                              min_overlap_part=min_overlap_part)
    widthRatio = width / div_side_length
    heightRatio = height / div_side_length
    # Counting total sum of predicted masks
    size = 0
    for res in results:
        size += len(res['scores'])

    # Initialisation of arrays
    if use_mini_mask:
        masks = np.zeros((div_side_length, div_side_length, size), dtype=bool)
    else:
        masks = np.zeros((height, width, size), dtype=bool)
    scores = np.zeros(size)
    rois = np.zeros((size, 4), dtype=int)
    class_ids = np.zeros(size, dtype=int)

    # Iterating through divisions results
    lastIndex = 0
    for res in results:
        # Getting the division ID based on iterator or given ids and getting its coordinates
        divId = res["div_id"]
        xStart, xEnd, yStart, yEnd = dD.getDivisionByID(xStarts, yStarts, divId, division_size)

        # Formatting and adding all the division's predictions to global ones
        for prediction_index in range(len(res['scores'])):
            scores[lastIndex] = res['scores'][prediction_index]
            class_ids[lastIndex] = res['class_ids'][prediction_index]

            if use_mini_mask:
                masks[:, :, lastIndex] = res['masks'][:, :, prediction_index]
            elif division_size != "noDiv":
                mask = res['masks'][:, :, prediction_index]
                if mask.shape[0] != division_size or mask.shape[1] != division_size:
                    tempWidthRatio = division_size / mask.shape[1]
                    tempHeightRatio = division_size / mask.shape[0]
                    roi = res['rois'][prediction_index]
                    roi[0] *= tempHeightRatio
                    roi[1] *= tempWidthRatio
                    roi[2] *= tempHeightRatio
                    roi[3] *= tempWidthRatio
                    mask = cv2.resize(np.uint8(mask), (division_size, division_size))
                masks[yStart:yEnd, xStart:xEnd, lastIndex] = mask
            else:
                mask = np.uint8(res['masks'][:, :, prediction_index])
                masks[:, :, lastIndex] = cv2.resize(mask, (width, height), interpolation=cv2.INTER_CUBIC)

            roi = res['rois'][prediction_index].copy()
            # y1, x1, y2, x2
            if division_size != "noDiv":
                roi[0] += yStart
                roi[1] += xStart
                roi[2] += yStart
                roi[3] += xStart
            else:
                roi[0] *= heightRatio
                roi[1] *= widthRatio
                roi[2] *= heightRatio
                roi[3] *= widthRatio
            rois[lastIndex] = roi

            lastIndex += 1

    # Formatting returned result
    fused_results = {
        "rois": rois,
        "class_ids": class_ids,
        "scores": scores,
        "masks": masks
    }
    return fused_results


def fuse_masks(fused_results, bb_threshold=0.1, mask_threshold=0.1, config: Config = None, displayProgress: str = None,
               verbose=0):
    """
    Fuses overlapping masks of the same class
    :param fused_results: the fused predictions results
    :param bb_threshold: least part of bounding boxes overlapping to continue checking
    :param mask_threshold: idem but with mask
    :param config: the config to get mini_mask informations
    :param displayProgress: if string given, prints a progress bar using this as prefix
    :param verbose: 0 : nothing, 1+ : errors/problems, 2 : general information
    :return: fused_results with only fused masks
    """
    rois = fused_results['rois']
    masks = fused_results['masks']
    scores = fused_results['scores']
    class_ids = fused_results['class_ids']

    if 'bbox_areas' in fused_results:
        bbAreas = fused_results['bbox_areas']
    else:
        bbAreas = np.ones(len(class_ids), dtype=int) * -1
    if 'mask_areas' in fused_results:
        maskAreas = fused_results['mask_areas']
    else:
        maskAreas = np.ones(len(class_ids), dtype=int) * -1
    fusedWith = np.ones(len(class_ids), dtype=int) * -1
    maskCount = np.ones(len(class_ids), dtype=int)

    toDelete = []
    if displayProgress is not None:
        total = combination(len(class_ids), 2)
        displayStep = max(round(total / 200), 1)
        current = 1
        start_time = time()
        duration = ""
        progressBar(0, total, prefix=displayProgress)
    for idxI, roi1 in enumerate(rois):
        # Computation of the bounding box area if not done yet
        if bbAreas[idxI] == -1:
            bbAreas[idxI] = get_bbox_area(roi1)

        for idxJ in range(idxI + 1, len(rois)):
            if displayProgress is not None:
                if current == total:
                    duration = f"Duration = {formatTime(round(time() - start_time))}"
                if current % displayStep == 0 or current == total:
                    progressBar(current, total, prefix=displayProgress, suffix=duration)
                current += 1
            '''###################################
            ###     CHECKING BBOX OVERLAP      ###
            ###################################'''
            # If masks are not from the same class or have been fused with same mask, we skip them
            if idxI == idxJ or class_ids[idxI] != class_ids[idxJ] \
                    or (fusedWith[idxI] == fusedWith[idxJ] and fusedWith[idxI] != -1):
                continue

            hadPrinted = False
            roi2 = rois[idxJ]

            # Computation of the bounding box area if not done yet
            if bbAreas[idxJ] == -1:
                bbAreas[idxJ] = get_bbox_area(roi2)

            # Computation of the bb intersection
            bbIntersection = get_bboxes_intersection(roi1, roi2)

            # We skip next part if bb intersection not representative enough
            partOfRoI1 = bbIntersection / bbAreas[idxI]
            partOfRoI2 = bbIntersection / bbAreas[idxJ]

            '''###################################
            ###     CHECKING MASK OVERLAP      ###
            ###################################'''
            if partOfRoI1 > bb_threshold or partOfRoI2 > bb_threshold:
                if verbose > 1:
                    hadPrinted = True
                    print("[{:03d}/{:03d}] Enough RoI overlap".format(idxI, idxJ))

                mask1 = masks[:, :, idxI]
                mask2 = masks[:, :, idxJ]

                if config is not None and config.USE_MINI_MASK:
                    mask1, mask2 = expand_masks(mask1, roi1, mask2, roi2)

                if maskAreas[idxI] == -1:
                    maskAreas[idxI], verbose_output = get_mask_area(mask1, verbose=verbose)
                    if verbose_output is not None:
                        hadPrinted = True
                        print("[{:03d}] {}".format(idxI, verbose_output))

                if maskAreas[idxJ] == -1:
                    maskAreas[idxJ], verbose_output = get_mask_area(mask2, verbose=verbose)
                    if verbose_output is not None:
                        hadPrinted = True
                        print("[{:03d}] {}".format(idxJ, verbose_output))

                # Computing intersection of mask 1 and 2 and computing its area
                mask1AND2 = np.logical_and(mask1, mask2)
                mask1AND2Area, _ = get_mask_area(mask1AND2, verbose=verbose)
                partOfMask1 = mask1AND2Area / maskAreas[idxI]
                partOfMask2 = mask1AND2Area / maskAreas[idxJ]

                if verbose > 0:
                    verbose_output = "[{:03d}] Intersection representing more than 100% of the mask : {:3.2f}%"
                    if not (0 <= partOfMask1 <= 1):
                        hadPrinted = True
                        print(verbose_output.format(idxI, partOfMask1 * 100))

                    if not (0 <= partOfMask2 <= 1):
                        hadPrinted = True
                        print(verbose_output.format(idxJ, partOfMask2 * 100))

                    if verbose > 1:
                        print("[OR] {:5.2f}% of mask [{:03d}]".format(partOfMask1 * 100, idxI))
                        print("[OR] {:5.2f}% of mask [{:03d}]".format(partOfMask2 * 100, idxJ))

                '''####################
                ###     FUSION      ###
                ####################'''
                if partOfMask1 > mask_threshold or partOfMask2 > mask_threshold:
                    # If the first mask has already been fused with another mask, we will fuse with the "parent" one
                    if fusedWith[idxI] == fusedWith[idxJ] == -1:  # No mask fused
                        receiver = idxI
                        giver = idxJ
                    elif fusedWith[idxI] != -1:  # I fused
                        if fusedWith[idxJ] == -1:  # I fused, not J
                            receiver = fusedWith[idxI]
                            giver = idxJ
                        else:  # I and J fused but not with each other (previous test)
                            receiver = min(fusedWith[idxI], fusedWith[idxJ])
                            giver = max(fusedWith[idxI], fusedWith[idxJ])
                            for idx in range(len(fusedWith)):  # As giver will be deleted, we have to update the list
                                if fusedWith[idx] == giver:
                                    fusedWith[idx] = receiver
                    else:  # J fused, not I (previous test)
                        receiver = fusedWith[idxJ]
                        giver = idxI

                    fusedWith[giver] = receiver
                    toDelete.append(giver)

                    if verbose > 1:
                        print("[{:03d}] Fusion with [{:03d}]".format(giver, receiver))

                    receiverMask = masks[:, :, receiver]
                    giverMask = masks[:, :, giver]
                    if config is not None and config.USE_MINI_MASK:
                        receiverRoI = rois[receiver]
                        giverRoI = rois[giver]
                        receiverMask, giverMask = expand_masks(receiverMask, receiverRoI, giverMask, giverRoI)
                    fusedMask = np.logical_or(receiverMask, giverMask)

                    if verbose > 1:
                        verbose_output = "[{idReceiver:03d}] Receiver Mask {when} fusion:\n"
                        verbose_output += "\tROI = {roi}\n"
                        verbose_output += "\tROI area = {roiArea}\n"
                        verbose_output += "\tMask area = {maskArea}\n"
                        verbose_output += "\tScore = {score}\n"
                        verbose_output += "\tMask count = {count}\n"
                        print(verbose_output.format(when="before", idReceiver=receiver, roi=rois[receiver],
                                                    roiArea=bbAreas[receiver], maskArea=maskAreas[receiver],
                                                    score=scores[receiver], count=maskCount[receiver]))

                    # Updating the receiver mask's infos
                    rois[receiver] = global_bbox(rois[receiver], rois[giver])
                    if config is not None and config.USE_MINI_MASK:
                        shifted_fusedRoI = shift_bbox(rois[receiver])
                        masks[:, :, receiver] = minimize_mask(shifted_fusedRoI, fusedMask, config.MINI_MASK_SHAPE)
                    else:
                        masks[:, :, receiver] = fusedMask
                    bbAreas[receiver] = get_bbox_area(rois[receiver])
                    maskAreas[receiver], _ = get_mask_area(fusedMask)
                    scores[receiver] = (scores[receiver] * maskCount[receiver] + scores[idxJ])
                    maskCount[receiver] += 1
                    scores[receiver] /= maskCount[receiver]

                    if verbose > 1:
                        print(verbose_output.format(when="after", idReceiver=receiver, roi=rois[receiver],
                                                    roiArea=bbAreas[receiver], maskArea=maskAreas[receiver],
                                                    score=scores[receiver], count=maskCount[receiver]))

                if verbose > 0 and hadPrinted:
                    print(flush=True)
    if displayProgress is not None and current <= total:
        duration = f"Duration = {formatTime(round(time() - start_time))}"
        progressBar(total, total, prefix=displayProgress, suffix=duration, forceNewLine=True)
    # Deletion of unwanted results
    scores = np.delete(scores, toDelete)
    class_ids = np.delete(class_ids, toDelete)
    bbAreas = np.delete(bbAreas, toDelete)
    maskAreas = np.delete(maskAreas, toDelete)
    masks = np.delete(masks, toDelete, axis=2)
    rois = np.delete(rois, toDelete, axis=0)
    return {"rois": rois, "bbox_areas": bbAreas, "class_ids": class_ids,
            "scores": scores, "masks": masks, "mask_areas": maskAreas}


def fuse_class(fused_results, bb_threshold=0.1, mask_threshold=0.1, classes_compatibility=None, config: Config = None,
               displayProgress: str = None, verbose=0):
    """
    Fuses overlapping masks of the different classes
    :param fused_results: the fused predictions results
    :param bb_threshold: least part of bounding boxes overlapping to continue checking
    :param mask_threshold: idem but with mask
    :param classes_compatibility:
    :param config: the config to get mini_mask informations
    :param displayProgress: if string given, prints a progress bar using this as prefix
    :param verbose: 0 : nothing, 1+ : errors/problems, 2 : general information
    :return: fused_results with only fused masks
    """

    if classes_compatibility is None or len(classes_compatibility) == 0:
        return fused_results

    rois = fused_results['rois']
    masks = fused_results['masks']
    scores = fused_results['scores']
    class_ids = fused_results['class_ids']
    indices = np.arange(len(class_ids))

    if 'bbox_areas' in fused_results:
        bbAreas = fused_results['bbox_areas']
    else:
        bbAreas = np.ones(len(class_ids), dtype=int) * -1
    if 'mask_areas' in fused_results:
        maskAreas = fused_results['mask_areas']
    else:
        maskAreas = np.ones(len(class_ids), dtype=int) * -1
    fusedWith = np.ones(len(class_ids), dtype=int) * -1
    maskCount = np.ones(len(class_ids), dtype=int)

    if displayProgress is not None:
        total = len(classes_compatibility)
        start_time = time()
        duration = ""

    toDelete = []
    # For each set of class, selecting the indices that corresponds
    for progressOffset, current_classes in enumerate(classes_compatibility):
        if type(current_classes) is int:
            _current_classes = [current_classes]
        else:
            _current_classes = current_classes
        current_indices = indices[np.isin(class_ids, _current_classes)]
        if displayProgress is not None:
            stepTotal = combination(len(current_indices), 2)
            displayStep = max(round(total / 200), 1)
            current = 1
            progressBar(progressOffset, total, prefix=displayProgress)
        for current_idx, idxI in enumerate(current_indices):
            roi1 = rois[idxI]
            # Computation of the bounding box area if not done yet
            if bbAreas[idxI] == -1:
                bbAreas[idxI] = get_bbox_area(roi1)

            for idxJ in current_indices[current_idx + 1:]:
                if displayProgress is not None:
                    if current == total:
                        duration = f"Duration = {formatTime(round(time() - start_time))}"
                    if current % displayStep == 0 or current == stepTotal:
                        progressBar(progressOffset + (current / stepTotal), total, prefix=displayProgress,
                                    suffix=duration)
                    current += 1
                '''###################################
                ###     CHECKING BBOX OVERLAP      ###
                ###################################'''
                # If masks are not from the same class or have been fused with same mask, we skip them
                if idxI == idxJ or (fusedWith[idxI] == fusedWith[idxJ] and fusedWith[idxI] != -1):
                    continue

                hadPrinted = False
                roi2 = rois[idxJ]

                # Computation of the bounding box area if not done yet
                if bbAreas[idxJ] == -1:
                    bbAreas[idxJ] = get_bbox_area(roi2)

                # Computation of the bb intersection
                bbIntersection = get_bboxes_intersection(roi1, roi2)

                # We skip next part if bb intersection not representative enough
                partOfRoI1 = bbIntersection / bbAreas[idxI]
                partOfRoI2 = bbIntersection / bbAreas[idxJ]

                '''###################################
                ###     CHECKING MASK OVERLAP      ###
                ###################################'''
                if partOfRoI1 > bb_threshold or partOfRoI2 > bb_threshold:
                    if verbose > 1:
                        hadPrinted = True
                        print("[{:03d}/{:03d}] Enough RoI overlap".format(idxI, idxJ))

                    mask1 = masks[:, :, idxI]
                    mask2 = masks[:, :, idxJ]

                    if config is not None and config.USE_MINI_MASK:
                        mask1, mask2 = expand_masks(mask1, roi1, mask2, roi2)

                    if maskAreas[idxI] == -1:
                        maskAreas[idxI], verbose_output = get_mask_area(mask1, verbose=verbose)
                        if verbose_output is not None:
                            hadPrinted = True
                            print("[{:03d}] {}".format(idxI, verbose_output))

                    if maskAreas[idxJ] == -1:
                        maskAreas[idxJ], verbose_output = get_mask_area(mask2, verbose=verbose)
                        if verbose_output is not None:
                            hadPrinted = True
                            print("[{:03d}] {}".format(idxJ, verbose_output))

                    # Computing intersection of mask 1 and 2 and computing its area
                    mask1AND2 = np.logical_and(mask1, mask2)
                    mask1AND2Area, _ = get_mask_area(mask1AND2, verbose=verbose)
                    partOfMask1 = mask1AND2Area / maskAreas[idxI]
                    partOfMask2 = mask1AND2Area / maskAreas[idxJ]

                    if verbose > 0:
                        verbose_output = "[{:03d}] Intersection representing more than 100% of the mask : {:3.2f}%"
                        if not (0 <= partOfMask1 <= 1):
                            hadPrinted = True
                            print(verbose_output.format(idxI, partOfMask1 * 100))

                        if not (0 <= partOfMask2 <= 1):
                            hadPrinted = True
                            print(verbose_output.format(idxJ, partOfMask2 * 100))

                        if verbose > 1:
                            print("[OR] {:5.2f}% of mask [{:03d}]".format(partOfMask1 * 100, idxI))
                            print("[OR] {:5.2f}% of mask [{:03d}]".format(partOfMask2 * 100, idxJ))

                    '''####################
                    ###     FUSION      ###
                    ####################'''
                    if partOfMask1 > mask_threshold or partOfMask2 > mask_threshold:
                        # If the first mask has already been fused with another mask, we will fuse with the "parent" one
                        if fusedWith[idxI] == fusedWith[idxJ] == -1:  # No mask fused
                            receiver = idxI
                            giver = idxJ
                        elif fusedWith[idxI] != -1:  # I fused
                            if fusedWith[idxJ] == -1:  # I fused, not J
                                receiver = fusedWith[idxI]
                                giver = idxJ
                            else:  # I and J fused but not with each other (previous test)
                                receiver = min(fusedWith[idxI], fusedWith[idxJ])
                                giver = max(fusedWith[idxI], fusedWith[idxJ])
                                for idx in range(
                                        len(fusedWith)):  # As giver will be deleted, we have to update the list
                                    if fusedWith[idx] == giver:
                                        fusedWith[idx] = receiver
                        else:  # J fused, not I (previous test)
                            receiver = fusedWith[idxJ]
                            giver = idxI

                        fusedWith[giver] = receiver
                        toDelete.append(giver)

                        if verbose > 1:
                            print("[{:03d}] Fusion with [{:03d}]".format(giver, receiver))

                        receiverMask = masks[:, :, receiver]
                        giverMask = masks[:, :, giver]
                        if config is not None and config.USE_MINI_MASK:
                            receiverRoI = rois[receiver]
                            giverRoI = rois[giver]
                            receiverMask, giverMask = expand_masks(receiverMask, receiverRoI, giverMask, giverRoI)
                        fusedMask = np.logical_or(receiverMask, giverMask)

                        if verbose > 1:
                            verbose_output = "[{idReceiver:03d}] Receiver Mask {when} fusion:\n"
                            verbose_output += "\tROI = {roi}\n"
                            verbose_output += "\tROI area = {roiArea}\n"
                            verbose_output += "\tMask area = {maskArea}\n"
                            verbose_output += "\tScore = {score}\n"
                            verbose_output += "\tClass = {class_id}\n"
                            verbose_output += "\tMask count = {count}\n"
                            print(verbose_output.format(when="before", idReceiver=receiver, roi=rois[receiver],
                                                        roiArea=bbAreas[receiver], maskArea=maskAreas[receiver],
                                                        score=scores[receiver], class_id=class_ids[receiver],
                                                        count=maskCount[receiver]))

                        # Updating the receiver mask's infos
                        # If we are fusing masks from different classes, we use the one with best score
                        if class_ids[giver] != class_ids[receiver] and scores[receiver] < scores[giver]:
                            class_ids[receiver] = class_ids[giver]
                        rois[receiver] = global_bbox(rois[receiver], rois[giver])
                        if config is not None and config.USE_MINI_MASK:
                            shifted_fusedRoI = shift_bbox(rois[receiver])
                            masks[:, :, receiver] = minimize_mask(shifted_fusedRoI, fusedMask, config.MINI_MASK_SHAPE)
                        else:
                            masks[:, :, receiver] = fusedMask
                        bbAreas[receiver] = get_bbox_area(rois[receiver])
                        maskAreas[receiver], _ = get_mask_area(fusedMask)
                        scores[receiver] = (scores[receiver] * maskCount[receiver] + scores[idxJ])
                        maskCount[receiver] += 1
                        scores[receiver] /= maskCount[receiver]

                        if verbose > 1:
                            print(verbose_output.format(when="after", idReceiver=receiver, roi=rois[receiver],
                                                        roiArea=bbAreas[receiver], maskArea=maskAreas[receiver],
                                                        score=scores[receiver], class_id=class_ids[receiver],
                                                        count=maskCount[receiver]))

                    if verbose > 0 and hadPrinted:
                        print(flush=True)
        if displayProgress is not None and duration == "":
            duration = f"Duration = {formatTime(round(time() - start_time))}"
            progressBar(1, 1, prefix=displayProgress, suffix=duration, forceNewLine=True)
        # Deletion of unwanted results
    scores = np.delete(scores, toDelete)
    class_ids = np.delete(class_ids, toDelete)
    bbAreas = np.delete(bbAreas, toDelete)
    maskAreas = np.delete(maskAreas, toDelete)
    masks = np.delete(masks, toDelete, axis=2)
    rois = np.delete(rois, toDelete, axis=0)
    return {"rois": rois, "bbox_areas": bbAreas, "class_ids": class_ids,
            "scores": scores, "masks": masks, "mask_areas": maskAreas}


class FilterBehavior(Enum):
    ERASED = -1
    BEST_CAN_BE_INCLUDED = 0
    BEST_NOT_INCLUDED = 3
    ERASE = 1
    KEEP = 2

    def opposite(self):
        if self.value in [-1, 1]:
            return FilterBehavior(self.value * -1)
        else:
            return FilterBehavior(self.value)


def comparePriority(class_id1, class_id2, priority_table=None, default: FilterBehavior = FilterBehavior.KEEP):
    """
    Compare priority of given class ids
    :param class_id1: the first class id
    :param class_id2: the second class id
    :param priority_table: the priority table to get the priority in
    :param default: default behavior if priority cannot be found
    :return: FilterBehavior value
    """
    # Return 0 if no priority table given, if it has bad dimensions or a class_id is not in the correct range
    if priority_table is None or not (len(priority_table) == len(priority_table[0]) and 0 <= class_id1 < len(
            priority_table) and 0 <= class_id2 < len(priority_table)) or type(priority_table[0][0]) not in [bool, int]:
        return default
    elif type(priority_table[0][0]) in [int, FilterBehavior]:
        res = priority_table[class_id1][class_id2]
        if type(res) is int:
            res = FilterBehavior(res)
        return res
    elif type(priority_table[0][0]) is bool:
        if priority_table[class_id1][class_id2]:
            return FilterBehavior.ERASE
        elif priority_table[class_id2][class_id1]:
            return FilterBehavior.ERASED
        else:
            return default
    else:
        return default


def filter_masks(fused_results, bb_threshold=0.5, mask_threshold=0.2, priority_table=None,
                 config: Config = None, displayProgress: str = None, verbose=0):
    """
    Post-prediction filtering to remove non-sense predictions
    :param fused_results: the results after fusion
    :param bb_threshold: the least part of overlapping bounding boxes to continue checking
    :param mask_threshold: the least part of a mask contained in another for it to be deleted
    :param priority_table: the priority table used to compare classes
    :param config: the config to get mini_mask informations
    :param displayProgress: if string given, prints a progress bar using this as prefix
    :param verbose: 0 : nothing, 1+ : errors/problems, 2 : general information
    :return:
    """
    if priority_table is None:
        return fused_results
    rois = fused_results['rois']
    masks = fused_results['masks']
    scores = fused_results['scores']
    class_ids = fused_results['class_ids']

    if 'bbox_areas' in fused_results:
        bbAreas = fused_results['bbox_areas']
    else:
        bbAreas = np.ones(len(class_ids), dtype=int) * -1
    if 'mask_areas' in fused_results:
        maskAreas = fused_results['mask_areas']
    else:
        maskAreas = np.ones(len(class_ids), dtype=int) * -1

    toDelete = []
    if displayProgress is not None:
        total = combination(len(class_ids), 2)
        displayStep = max(round(total / 200), 1)
        current = 1
        start_time = time()
        duration = ""
        progressBar(0, total, prefix=displayProgress)
    for i, roi1 in enumerate(rois):
        # If this RoI has already been selected for deletion, we skip it
        if i in toDelete:
            continue

        # If the area of this RoI has not been computed
        if bbAreas[i] == -1:
            bbAreas[i] = get_bbox_area(roi1)

        # Then we check for each RoI that has not already been checked
        for j in range(i + 1, len(rois)):
            if displayProgress is not None:
                if current == total:
                    duration = f"Duration = {formatTime(round(time() - start_time))}"
                if current % displayStep == 0 or current == total:
                    progressBar(current, total, prefix=displayProgress, suffix=duration)
                current += 1
            if j in toDelete:
                continue
            roi2 = rois[j]

            # We want only one prediction class to be vessel
            priority = comparePriority(class_ids[i] - 1, class_ids[j] - 1, priority_table)
            if priority == FilterBehavior.KEEP:
                continue

            # If the area of the 2nd RoI has not been computed
            if bbAreas[j] == -1:
                bbAreas[j] = get_bbox_area(roi2)

            # Computation of the bb intersection
            intersection = get_bboxes_intersection(roi1, roi2)

            # We skip next part if bb intersection not representative enough
            partOfR1 = intersection / bbAreas[i]
            partOfR2 = intersection / bbAreas[j]
            if partOfR1 > bb_threshold or partOfR2 > bb_threshold:
                # Getting first mask and computing its area if not done yet
                mask1 = masks[:, :, i]
                mask2 = masks[:, :, j]

                if config is not None and config.USE_MINI_MASK:
                    mask1, mask2 = expand_masks(mask1, roi1, mask2, roi2)

                if maskAreas[i] == -1:
                    maskAreas[i], _ = get_mask_area(mask1, verbose=verbose)
                    if maskAreas[i] == 0:
                        print(i, maskAreas[i])

                # Getting second mask and computing its area if not done yet
                if maskAreas[j] == -1:
                    maskAreas[j], _ = get_mask_area(mask2, verbose=verbose)
                    if maskAreas[j] == 0:
                        print(j, maskAreas[j])

                # Computing intersection of mask 1 and 2 and computing its area
                mask1AND2 = np.logical_and(mask1, mask2)
                mask1AND2Area, _ = get_mask_area(mask1AND2, verbose=verbose)
                partOfMask1 = mask1AND2Area / maskAreas[i]
                partOfMask2 = mask1AND2Area / maskAreas[j]

                # We check if the common area represents more than the mask_threshold
                if partOfMask1 > mask_threshold or partOfMask2 > mask_threshold:
                    if priority == FilterBehavior.ERASED and partOfMask1 > mask_threshold:
                        if verbose > 0:
                            print(f"[{i:03d}/{j:03d}] Kept class = {class_ids[j]}\tRemoved Class = {class_ids[i]}")
                        toDelete.append(i)
                    elif priority == FilterBehavior.ERASE and partOfMask2 > mask_threshold:
                        if verbose > 0:
                            print(f"[{i:03d}/{j:03d}] Kept class = {class_ids[i]}\tRemoved Class = {class_ids[j]}")
                        toDelete.append(j)
                    elif priority == FilterBehavior.BEST_CAN_BE_INCLUDED:
                        worst, best = (j, i) if scores[i] > scores[j] else (i, j)
                        if verbose > 0:
                            print(
                                f"[{i:03d}/{j:03d}] Kept class = {class_ids[worst]}\tRemoved Class = {class_ids[best]}")
                        toDelete.append(worst)
                    elif priority == FilterBehavior.BEST_NOT_INCLUDED:
                        # if partOfMask1 > included_threshold and partOfMask2 <= including_threshold:
                        if partOfMask1 > mask_threshold > partOfMask2:
                            worst, best = (i, j)
                        # elif partOfMask2 > included_threshold and partOfMask1 <= including_threshold:
                        elif partOfMask1 < mask_threshold < partOfMask2:
                            worst, best = (j, i)
                        else:
                            worst, best = (j, i) if scores[i] > scores[j] else (i, j)
                        if verbose > 0:
                            print(
                                f"[{i:03d}/{j:03d}] Kept class = {class_ids[worst]}\tRemoved Class = {class_ids[best]}")
                        toDelete.append(worst)
    if displayProgress is not None and current <= total:
        duration = f"Duration = {formatTime(round(time() - start_time))}"
        progressBar(total, total, prefix=displayProgress, suffix=duration, forceNewLine=True)
    # Deletion of unwanted results
    scores = np.delete(scores, toDelete)
    class_ids = np.delete(class_ids, toDelete)
    bbAreas = np.delete(bbAreas, toDelete)
    maskAreas = np.delete(maskAreas, toDelete)
    masks = np.delete(masks, toDelete, axis=2)
    rois = np.delete(rois, toDelete, axis=0)
    return {"rois": rois, "bbox_areas": bbAreas, "class_ids": class_ids,
            "scores": scores, "masks": masks, "mask_areas": maskAreas}


def filter_orphan_masks(results, bb_threshold=0.5, mask_threshold=0.5, classes_hierarchy=None,
                        config: Config = None, displayProgress: str = None, verbose=0):
    """
    Post-prediction filtering to remove non-sense predictions
    :param results: the results after fusion
    :param bb_threshold: the least part of overlapping bounding boxes to continue checking
    :param mask_threshold: the least part of a mask contained in another for it to be kept
    :param classes_hierarchy: the parents/children classes hierarchy to use to filter masks
                              {id_class : {"contains": [id_class], "keep_if_no_child": bool}}
    :param config: the config to get mini_mask informations
    :param displayProgress: if string given, prints a progress bar using this as prefix
    :param verbose: 0 : nothing, 1+ : errors/problems, 2 : general information
    :return:
    """
    if classes_hierarchy is None or len(classes_hierarchy) == 0:
        return results
    rois = results['rois']
    masks = results['masks']
    scores = results['scores']
    class_ids = results['class_ids']
    bbAreas = results['bbox_areas']
    maskAreas = results['mask_areas']
    indices = np.arange(len(class_ids))
    if displayProgress is not None:
        total = len(classes_hierarchy)
        current = 0
        start_time = time()
        duration = ""
    toDelete = []
    for parentClass in classes_hierarchy:
        if displayProgress is not None:
            progressBar(current, total, prefix=displayProgress)

        # Getting mask ids from the parent class and children classes
        parentClassIds = indices[np.isin(class_ids, [parentClass])]
        childClassIds = indices[np.isin(class_ids, classes_hierarchy[parentClass]["contains"])]

        # Initializing list of masks that will be deleted if they are not matching a parent/child masks
        toDeleteClass = childClassIds.tolist()
        if not classes_hierarchy[parentClass]["keep_if_no_child"]:
            toDeleteClass.extend(parentClassIds.tolist())

        if verbose > 1:  # masks with no parents at all
            print(f"\nChecking parent/child classes : ({parentClass}, {classes_hierarchy[parentClass]['contains']})")

        # If there is something to check : at least one parent and one child
        if len(childClassIds) > 0 and len(parentClassIds) > 0:
            parentTotalStep = len(childClassIds) * len(parentClassIds)
            displayStep = max(round(parentTotalStep / 20), 1)
            iterator = 0

            # For each parent mask we will test if a child class's mask is overlapping
            for parentId in parentClassIds:
                roi1 = rois[parentId]
                # If the area of this RoI has not been computed
                if bbAreas[parentId] == -1:
                    bbAreas[parentId] = get_bbox_area(roi1)
                # Then we check for each RoI that has not already been checked
                parentMaskHasChild = classes_hierarchy[parentClass]["keep_if_no_child"]
                for childId in childClassIds:
                    if displayProgress is not None:
                        if iterator % displayStep == 0:
                            progress = current + iterator / parentTotalStep
                            if verbose > 3:
                                print("\nProgress =", progressText(progress, total))
                            progressBar(progress, total, prefix=displayProgress, suffix=duration)
                    if childId not in toDeleteClass:
                        continue
                    roi2 = rois[childId]

                    # If the area of the 2nd RoI has not been computed
                    if bbAreas[childId] == -1:
                        bbAreas[childId] = get_bbox_area(roi2)

                    # Computation of the bb intersection
                    intersection = get_bboxes_intersection(roi1, roi2)

                    # We skip next part if bb intersection not representative enough
                    partOfR1 = intersection / bbAreas[parentId]
                    partOfR2 = intersection / bbAreas[childId]
                    if partOfR1 > bb_threshold or partOfR2 > bb_threshold:
                        # Getting first mask and computing its area if not done yet
                        mask1 = masks[:, :, parentId]
                        mask2 = masks[:, :, childId]

                        if config is not None and config.USE_MINI_MASK:
                            mask1, mask2 = expand_masks(mask1, roi1, mask2, roi2)

                        # Getting second mask and computing its area if not done yet
                        if maskAreas[childId] == -1:
                            maskAreas[childId], _ = get_mask_area(mask2, verbose=verbose)
                            if maskAreas[childId] == 0:
                                print(childId, maskAreas[childId])

                        # Computing intersection of mask 1 and 2 and computing its area
                        mask1AND2 = np.logical_and(mask1, mask2)
                        mask1AND2Area, _ = get_mask_area(mask1AND2, verbose=verbose)
                        partOfMask2 = mask1AND2Area / maskAreas[childId]

                        # We check if the common area represents more than the vessel_threshold of the non-vessel mask
                        if partOfMask2 > mask_threshold:
                            parentMaskHasChild = True
                            if verbose > 2:
                                print(f"\nmask {childId} removed from orphans")
                            toDeleteClass.remove(childId)
                    iterator += 1
                # If parent mask has a child and it was required, we remove it from the list of ids to delete
                if not classes_hierarchy[parentClass]["keep_if_no_child"] and parentMaskHasChild:
                    try:
                        toDeleteClass.remove(parentId)
                    except ValueError:
                        if verbose > 1:
                            print(
                                f"\nTried to remove a parent mask ({parentId}) from the deletion list that not in it.")
        if displayProgress is not None:
            current += 1
        if verbose > 1:
            classes = []
            for id_ in toDeleteClass:
                if class_ids[id_] not in classes:
                    classes.append(int(class_ids[id_]))
            print(f"\nDeleting {len(toDeleteClass)} sole parent/orphan mask(s) from classes {classes}")
        toDelete.extend(toDeleteClass)
    # Deletion of unwanted results
    scores = np.delete(scores, toDelete)
    class_ids = np.delete(class_ids, toDelete)
    bbAreas = np.delete(bbAreas, toDelete)
    maskAreas = np.delete(maskAreas, toDelete)
    masks = np.delete(masks, toDelete, axis=2)
    rois = np.delete(rois, toDelete, axis=0)
    if displayProgress is not None and duration == "":
        duration = f"Duration = {formatTime(round(time() - start_time))}"
        progressBar(2, 2, prefix=displayProgress, suffix=duration, forceNewLine=True)

    return {"rois": rois, "bbox_areas": bbAreas, "class_ids": class_ids,
            "scores": scores, "masks": masks, "mask_areas": maskAreas}


def filter_small_masks(fused_results, min_size=300, classes: List[int] = None, config: Config = None,
                       displayProgress: str = None, verbose=0):
    """
    Post-prediction filtering to remove masks that are too small
    :param fused_results: the results after fusion
    :param min_size: the least area that a mask is allowed to have
    :param classes: list of classes ids to check, if None, all classes will be checked
    :param config: the config to get mini_mask informations
    :param displayProgress: if string given, prints a progress bar using this as prefix
    :param verbose: 0 : nothing, 1+ : errors/problems, 2 : general information
    :return:
    """
    rois = fused_results['rois']
    masks = fused_results['masks']
    scores = fused_results['scores']
    class_ids = fused_results['class_ids']

    if 'bbox_areas' in fused_results:
        bbAreas = fused_results['bbox_areas']
    else:
        bbAreas = np.ones(len(class_ids), dtype=int) * -1
    if 'mask_areas' in fused_results:
        maskAreas = fused_results['mask_areas']
    else:
        maskAreas = np.ones(len(class_ids), dtype=int) * -1

    toDelete = []
    if displayProgress is not None:
        total = len(class_ids)
        displayStep = max(round(total / 200), 1)
        start_time = time()
        duration = ""
        progressBar(0, total, prefix=displayProgress)

    for idx, roi in enumerate(rois):
        # If the mask class has to be checked
        if classes is None or class_ids[idx] in classes:
            # If the area of this RoI has not been computed
            if bbAreas[idx] == -1:
                bbAreas[idx] = get_bbox_area(roi)

            if bbAreas[idx] >= min_size:
                mask = masks[:, :, idx]

                if config is not None and config.USE_MINI_MASK:
                    shifted_roi = shift_bbox(rois[idx])
                    mask = expand_mask(shifted_roi, masks[:, :, idx], shifted_roi[2:])

                if maskAreas[idx] == -1:
                    maskAreas[idx], _ = get_mask_area(mask, verbose=verbose)

                if maskAreas[idx] < min_size:
                    toDelete.append(idx)
            else:
                toDelete.append(idx)

        if displayProgress is not None:
            _idx = idx + 1
            if _idx == total:
                duration = f"Duration = {formatTime(round(time() - start_time))}"
            if idx % displayStep == 0 or _idx == total:
                progressBar(_idx, total, prefix=displayProgress, suffix=duration)

    # Deletion of unwanted results
    scores = np.delete(scores, toDelete)
    class_ids = np.delete(class_ids, toDelete)
    bbAreas = np.delete(bbAreas, toDelete)
    maskAreas = np.delete(maskAreas, toDelete)
    masks = np.delete(masks, toDelete, axis=2)
    rois = np.delete(rois, toDelete, axis=0)
    return {"rois": rois, "bbox_areas": bbAreas, "class_ids": class_ids,
            "scores": scores, "masks": masks, "mask_areas": maskAreas}


def compute_on_border_part(image: np.ndarray, mask: np.ndarray):
    """
    Return part of mask not being on image as a float
    :param image: the RGB image on which the mask is applied
    :param mask: the mask to test with the same shape as image
    :return: part of the mask not being on the image as float
    """
    maskArea = dD.getBWCount(mask)[1]
    if maskArea == 0:  # If no mask
        return 1.
    # Converting the image to grayscale as it is needed by cv2.countNonZero() and avoiding computing on 3 channels
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    masked_image = cv2.bitwise_and(image, mask.astype(np.uint8) * 255)
    # Computing part of the mask that is not black pixels
    on_image_area = cv2.countNonZero(masked_image)
    return round(1. - on_image_area / maskArea, 2)


def filter_on_border_masks(fused_results, image: np.ndarray, onBorderThreshold=0.25, classes: List[int] = None,
                           config: Config = None, displayProgress: str = None, verbose=0):
    """
    Post-prediction filtering to remove masks that are too small
    :param fused_results: the results after fusion
    :param image: the image to check if the mask is on the border or not
    :param onBorderThreshold: the least part of a mask to be on the border/void (#000000 color) part of the image for it
                              to be deleted
    :param classes: list of classes ids to check, if None, all classes will be checked
    :param config: the config to get mini_mask informations
    :param displayProgress: if string given, prints a progress bar using this as prefix
    :param verbose: 0 : nothing, 1+ : errors/problems, 2 : general information
    :return:
    """
    rois = fused_results['rois']
    masks = fused_results['masks']
    scores = fused_results['scores']
    class_ids = fused_results['class_ids']
    indices = np.arange(len(class_ids))

    if 'bbox_areas' in fused_results:
        bbAreas = fused_results['bbox_areas']
    else:
        bbAreas = np.ones(len(class_ids), dtype=int) * -1
    if 'mask_areas' in fused_results:
        maskAreas = fused_results['mask_areas']
    else:
        maskAreas = np.ones(len(class_ids), dtype=int) * -1

    if classes is not None:
        if type(classes) is int:
            _classes = [classes]
        else:
            _classes = classes
        indices = indices[np.isin(class_ids, _classes)]

    toDelete = []
    if displayProgress is not None:
        total = len(indices)
        displayStep = max(round(total / 200), 1)
        start_time = time()
        duration = ""
        progressBar(0, total, prefix=displayProgress)

    for iterator, idx in enumerate(indices):
        roi = rois[idx]

        if config is not None and config.USE_MINI_MASK:
            shifted_roi = shift_bbox(roi)
            mask = expand_mask(shifted_roi, masks[:, :, idx], shifted_roi[2:])
        else:
            mask = masks[roi[0]:roi[2], roi[1]:roi[3], idx]

        imagePart = image[roi[0]:roi[2], roi[1]:roi[3], :]

        if maskAreas[idx] == -1:
            maskAreas[idx], _ = get_mask_area(mask, verbose=verbose)

        if compute_on_border_part(imagePart, mask) > onBorderThreshold:
            toDelete.append(idx)

        if displayProgress is not None:
            iterator = iterator + 1
            if iterator == total:
                duration = f"Duration = {formatTime(round(time() - start_time))}"
            if idx % displayStep == 0 or iterator == total:
                progressBar(iterator, total, prefix=displayProgress, suffix=duration)

    # Deletion of unwanted results
    scores = np.delete(scores, toDelete)
    class_ids = np.delete(class_ids, toDelete)
    bbAreas = np.delete(bbAreas, toDelete)
    maskAreas = np.delete(maskAreas, toDelete)
    masks = np.delete(masks, toDelete, axis=2)
    rois = np.delete(rois, toDelete, axis=0)
    return {"rois": rois, "bbox_areas": bbAreas, "class_ids": class_ids,
            "scores": scores, "masks": masks, "mask_areas": maskAreas}


def getCountAndArea(results: dict, classes: List[str], config=None):
    """
    Computing count and area of classes from results
    :param results: the results
    :param classes: list of classes' names
    :param config: the config to get mini_mask informations
    :return: Dict of "className": {"count": int, "area": int} elements for each classes
    """
    res = {}

    rois = results['rois']
    masks = results['masks']
    class_ids = results['class_ids']

    # Getting the inferenceIDs of the wanted classes
    selectedClassesID = {}
    for idx, className in enumerate(classes):
        selectedClassesID[idx + 1] = className
        res[className] = {"count": 0, "area": 0}

    # For each predictions, if class ID matching with one we want
    for index, classID in enumerate(class_ids):
        if classID in selectedClassesID.keys():
            # Getting current values of count and area
            className = selectedClassesID[classID]
            res[className]["count"] += 1
            # Getting the area of current mask
            if config is not None and config.USE_MINI_MASK:
                shifted_roi = shift_bbox(rois[index])
                mask = expand_mask(shifted_roi, masks[:, :, index], shifted_roi[2:])
            else:
                yStart, xStart, yEnd, xEnd = rois[index]
                mask = masks[yStart:yEnd, xStart:xEnd, index]
            mask = mask.astype(np.uint8)
            if "mask_areas" in results and results['mask_areas'][index] != -1:
                area = int(results['mask_areas'][index])
            else:
                area, _ = get_mask_area(mask)
            res[className]["area"] += area  # Cast to int to avoid "json 'int64' not serializable"

    return res
