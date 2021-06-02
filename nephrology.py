"""
Skinet (Segmentation of the Kidney through a Neural nETwork) Project

Copyright (c) 2021 Skinet Team
Licensed under the MIT License (see LICENSE for details)
Written by Adrien JAUGEY
"""
import gc
import json
import os
import shutil
import traceback
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from common_utils import progressBar, formatTime, formatDate, progressText
    from mrcnn.datasetDivider import CV2_IMWRITE_PARAM
    import time
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    from time import time
    from skimage.io import imsave
    from mrcnn import datasetDivider as dD

    from mrcnn.config import Config
    from mrcnn import utils
    from mrcnn import model as modellib
    from mrcnn import visualize
    from mrcnn import post_processing as pp


def get_ax(rows=1, cols=1, size=8):
    return plt.subplots(rows, cols, figsize=(size * cols, size * rows), frameon=False)


def listAvailableImage(dirPath: str):
    files = os.listdir(dirPath)
    image = []
    for file in files:
        extension = file.split('.')[-1]
        if extension in ['png', 'jpg']:
            image.append(file)

    for i in range(len(image)):
        image[i] = os.path.join(dirPath, image[i])
    return image


class NephrologyInferenceModel:

    def __init__(self, mode: str = "cortex", min_confidence=None, divisionSize=1024,
                 min_overlap_part_main=0.33, min_overlap_part_cortex=0.5, cortex_size=None,
                 mini_mask_size=256, forceFullSizeMasks=False, low_memory=False):
        print("Initialisation")
        if mode not in ['cortex', 'main']:
            mode = 'main'
        self.__CLASSES_INFO = [
            {"name": "cortex", "ignore": mode == "main"},
            {"name": "medulla", "ignore": mode == "main"},
            {"name": "fibrous_capsule", "ignore": mode == "main"},
            {"name": "non_atrophic_tubule", "ignore": mode == "cortex"},
            {"name": "atrophic_tubule", "ignore": mode == "cortex"},
            {"name": "nsg", "ignore": mode == "cortex"},
            {"name": "complete_glomeruli", "ignore": mode == "cortex"},
            {"name": "partial_glomeruli", "ignore": mode == "cortex"},
            {"name": "globally_sclerosis_glomeruli", "ignore": mode == "cortex"},
            {"name": "vein", "ignore": mode == "cortex"},
            {"name": "artery", "ignore": mode == "cortex"},
            {"name": "internal_elastic_lamina", "ignore": mode == "cortex"},
            {"name": "external_elastic_lamina", "ignore": mode == "cortex"}
        ]
        cortex_mode = mode == "cortex"
        self.__CORTEX_MODE = cortex_mode
        self.__MODEL_PATH = f"skinet_{mode}.h5"
        self.__DIVISION_SIZE = divisionSize
        self.__MIN_OVERLAP_PART_MAIN = min_overlap_part_main
        self.__MIN_OVERLAP_PART_CORTEX = min_overlap_part_cortex
        self.__MIN_OVERLAP_PART = min_overlap_part_cortex if self.__CORTEX_MODE else min_overlap_part_main
        self.__CORTEX_SIZE = None if not self.__CORTEX_MODE else (2048, 2048) if cortex_size is None else cortex_size
        self.__LOW_MEMORY = low_memory
        self.__CUSTOM_CLASS_NAMES = []
        for classInfo in self.__CLASSES_INFO:
            if not classInfo["ignore"]:
                self.__CUSTOM_CLASS_NAMES.append(classInfo["name"])
        self.__NB_CLASS = len(self.__CUSTOM_CLASS_NAMES)
        # Root directory of the project
        self.__ROOT_DIR = os.getcwd()

        # Directory to save logs and trained model
        self.__MODEL_DIR = os.path.join(self.__ROOT_DIR, "logs")

        # Configurations
        nbClass = self.__NB_CLASS
        divSize = 1024 if self.__DIVISION_SIZE == "noDiv" else self.__DIVISION_SIZE
        if min_confidence is None:
            min_confidence = 0.7 if cortex_mode else 0.5

        class SkinetConfig(Config):
            NAME = "skinet"
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            NUM_CLASSES = 1 + nbClass
            IMAGE_MIN_DIM = divSize
            IMAGE_MAX_DIM = divSize
            RPN_ANCHOR_SCALES = (8, 16, 64, 128, 256)
            TRAIN_ROIS_PER_IMAGE = 800
            DETECTION_MIN_CONFIDENCE = min_confidence
            STEPS_PER_EPOCH = 400
            VALIDATION_STEPS = 50
            USE_MINI_MASK = not cortex_mode and not forceFullSizeMasks
            MINI_MASK_SHAPE = (mini_mask_size, mini_mask_size)  # (height, width) of the mini-mask

        self.__CONFIG = SkinetConfig()

        # Recreate the model in inference mode
        self.__MODEL = modellib.MaskRCNN(mode="inference", config=self.__CONFIG, model_dir=self.__MODEL_DIR)

        # Load trained weights (fill in path to trained weights here)
        if not os.path.exists(self.__MODEL_PATH):
            utils.download_trained_weights(1)
        self.__MODEL.load_weights(self.__MODEL_PATH, by_name=True)
        print()

    def prepare_image(self, imagePath, results_path):
        """
        Creating png version if not existing and get some information
        :param imagePath: path to the image to use
        :param results_path: path to the results dir to create the image folder and paste it in
        :return: image, imageInfo = {"PATH": str, "DIR_PATH": str, "FILE_NAME": str, "NAME": str, "HEIGHT": int,
        "WIDTH": int, "NB_DIV": int, "X_STARTS": v, "Y_STARTS": list}
        """
        image = None
        fullImage = None
        imageInfo = None
        image_results_path = None
        if os.path.exists(imagePath):
            imageInfo = {
                'PATH': imagePath,
                'DIR_PATH': os.path.dirname(imagePath),
                'FILE_NAME': os.path.basename(imagePath)
            }
            imageInfo['NAME'] = imageInfo['FILE_NAME'].split('.')[0]
            imageInfo['IMAGE_FORMAT'] = imageInfo['FILE_NAME'].split('.')[-1]

            # Reading input image in RGB color order
            imageChanged = False
            if self.__CORTEX_MODE:  # If in cortex mode, resize image to lower resolution
                imageInfo['FULL_RES_IMAGE'] = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)
                height, width, _ = imageInfo['FULL_RES_IMAGE'].shape
                fullImage = cv2.resize(imageInfo['FULL_RES_IMAGE'], self.__CORTEX_SIZE)
                imageChanged = True
            else:
                fullImage = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)
                height, width, _ = fullImage.shape
            imageInfo['HEIGHT'] = int(height)
            imageInfo['WIDTH'] = int(width)

            # Conversion of the image if format is not png or jpg
            if imageInfo['IMAGE_FORMAT'] not in ['png', 'jpg']:
                imageInfo['IMAGE_FORMAT'] = 'jpg'
                imageChanged = True
                tempPath = os.path.join(imageInfo['PATH'], f"{imageInfo['NAME']}.{imageInfo['IMAGE_FORMAT']}")
                imsave(tempPath, fullImage)
                imageInfo['PATH'] = tempPath

            # Creating the result dir if given and copying the base image in it
            if results_path is not None:
                image_results_path = os.path.join(os.path.normpath(results_path), imageInfo['NAME'])
                os.makedirs(image_results_path, exist_ok=True)
                imageInfo['PATH'] = os.path.join(image_results_path, f"{imageInfo['NAME']}.{imageInfo['IMAGE_FORMAT']}")
                if not imageChanged:
                    shutil.copy2(imagePath, imageInfo['PATH'])
                else:
                    imsave(imageInfo['PATH'], fullImage)
            else:
                image_results_path = None

            # Computing divisions coordinates if needed and total number of div
            if self.__DIVISION_SIZE == "noDiv":
                imageInfo['X_STARTS'] = imageInfo['Y_STARTS'] = [0]
            else:
                imageInfo['X_STARTS'] = dD.computeStartsOfInterval(
                    maxVal=self.__CORTEX_SIZE[0] if self.__CORTEX_MODE else width,
                    intervalLength=self.__DIVISION_SIZE,
                    min_overlap_part=self.__MIN_OVERLAP_PART
                )
                imageInfo['Y_STARTS'] = dD.computeStartsOfInterval(
                    maxVal=self.__CORTEX_SIZE[1] if self.__CORTEX_MODE else height,
                    intervalLength=self.__DIVISION_SIZE,
                    min_overlap_part=self.__MIN_OVERLAP_PART
                )
            imageInfo['NB_DIV'] = dD.getDivisionsCount(imageInfo['X_STARTS'], imageInfo['Y_STARTS'])

        return image, fullImage, imageInfo, image_results_path

    def init_results_dir(self, results_path):
        if results_path is None or results_path in ['', '.', './', "/"]:
            lastDir = "results"
            remainingPath = ""
        else:
            results_path = os.path.normpath(results_path)
            lastDir = os.path.basename(results_path)
            remainingPath = os.path.dirname(results_path)
        results_path = os.path.normpath(os.path.join(remainingPath, f"{lastDir}_{formatDate()}"))
        os.makedirs(results_path)
        print(f"Results will be saved to {results_path}")
        logsPath = os.path.join(results_path, 'inference_data.csv')
        with open(logsPath, 'w') as results_log:
            results_log.write(f"Image; Duration (s); Precision; {os.path.basename(self.__MODEL_PATH)}\n")
        return results_path, logsPath

    def inference(self, images: list, results_path=None, save_results=True,
                  fusion_bb_threshold=0.1, fusion_mask_threshold=0.1,
                  filter_bb_threshold=0.3, filter_mask_threshold=0.3,
                  priority_table=None, displayOnlyStats=False, allowSparse=False,
                  minMaskArea=300, on_border_threshold=0.25):

        if len(images) == 0:
            print("Images list is empty, no inference to perform.")
            return

        # If results have to be saved, setting the results path and creating directory
        if save_results:
            results_path, logsPath = self.init_results_dir(results_path)
        else:
            print("No result will be saved")
            results_path = None

        if not self.__CORTEX_MODE and priority_table is None:
            #                  nAtro  atro    nsg   compG  partG  scNsg  vein  artery intLam  extLam
            priority_table = [[False, True, False, False, False, True, True, False, False, False], # non_atrophic_tubule
                              [False, False, False, False, False, True, True, False, False, False], # atrophic_tubule
                              [True, True, False, False, False, True, True, True, False, False], # nsg
                              [False, False, False, False, False, False, False, False, False, False], # complete_glomeruli
                              [False, False, False, False, False, False, False, False, False, False],
                              # partial_glomeruli
                              [True, True, False, False, False, False, False, False, False, False],
                              # globally_sclerosis_glomeruli
                              [False, False, False, False, False, False, False, False, False, False],  # vein
                              [True, True, False, False, False, True, True, False, False, False],  # artery
                              [False, False, False, False, False, False, False, False, False, False],
                              # internal_elastic_lamina
                              [False, False, False, False, False, False, False, False, False, False]]  # external_elastic_lamina

        total_start_time = time()
        failedImages = []
        for i, IMAGE_PATH in enumerate(images):
            try:
                start_time = time()
                print(f"Using {IMAGE_PATH} image file {progressText(i + 1, len(images))}")
                visualizeNames = self.__CUSTOM_CLASS_NAMES.copy()
                visualizeNames.insert(0, 'background')

                step = "image preparation"
                image, fullImage, imageInfo, image_results_path = self.prepare_image(IMAGE_PATH, results_path)
                # Getting predictions for each division

                res = []
                total_px = self.__CONFIG.IMAGE_MAX_DIM * self.__CONFIG.IMAGE_MIN_DIM
                skipped = 0
                debugIterator = -1
                skippedText = ""
                inference_start_time = time()
                if not displayOnlyStats:
                    progressBar(0, imageInfo["NB_DIV"], prefix=' - Inference')
                for divId in range(imageInfo["NB_DIV"]):
                    step = f"{divId} div processing"
                    division = dD.getImageDivision(fullImage if image is None else image, imageInfo["X_STARTS"],
                                                   imageInfo["Y_STARTS"], divId)
                    grayDivision = cv2.cvtColor(division, cv2.COLOR_RGB2GRAY)
                    colorPx = cv2.countNonZero(grayDivision)
                    del grayDivision
                    if colorPx / total_px > 0.1:
                        step = f"{divId} div inference"
                        results = self.__MODEL.detect([division])
                        results[0]["div_id"] = divId
                        if self.__CONFIG.USE_MINI_MASK:
                            res.append(utils.reduce_memory(results[0].copy(), config=self.__CONFIG,
                                                           allow_sparse=allowSparse))
                        else:
                            res.append(results[0].copy())
                        del results
                    elif not displayOnlyStats:
                        skipped += 1
                        skippedText = f"({skipped} empty division{'s' if skipped > 1 else ''} skipped) "
                    del division
                    gc.collect()
                    if not displayOnlyStats:
                        if divId + 1 == imageInfo["NB_DIV"]:
                            inference_duration = round(time() - inference_start_time)
                            skippedText += f"Duration = {formatTime(inference_duration)}"
                        progressBar(divId + 1, imageInfo["NB_DIV"], prefix=' - Inference', suffix=skippedText)

                # Post-processing of the predictions
                if not displayOnlyStats:
                    print(" - Fusing results of all divisions")

                step = "fusing results"
                res = pp.fuse_results(res, fullImage.shape, division_size=self.__DIVISION_SIZE,
                                      min_overlap_part=self.__MIN_OVERLAP_PART)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    step = "fusing masks"
                    progressBarPrefix = " - Fusing overlapping masks" if not displayOnlyStats else None
                    res = pp.fuse_masks(res, bb_threshold=fusion_bb_threshold, mask_threshold=fusion_mask_threshold,
                                        config=self.__CONFIG, displayProgress=progressBarPrefix, verbose=0)

                    if not self.__CORTEX_MODE and len(self.__CUSTOM_CLASS_NAMES) == 10:
                        step = "removing border masks"
                        progressBarPrefix = " - Removing border masks" if not displayOnlyStats else None
                        classes_to_check = [7, 8, 9, 10]
                        res = pp.filter_on_border_masks(res, fullImage if image is None else image,
                                                        onBorderThreshold=on_border_threshold,
                                                        classes=classes_to_check, config=self.__CONFIG,
                                                        displayProgress=progressBarPrefix, verbose=0)

                        # TODO : Build automatically classes_hierarchy

                        classes_hierarchy = {
                            3: {"contains": [4, 5], "keep_if_no_child": False},
                            8: {"contains": [9, 10], "keep_if_no_child": True}
                        }
                        step = "filtering orphan masks (pass 1)"
                        progressBarPrefix = " - Removing orphan masks" if not displayOnlyStats else None
                        res = pp.filter_orphan_masks(res, bb_threshold=filter_bb_threshold,
                                                     mask_threshold=filter_mask_threshold,
                                                     classes_hierarchy=classes_hierarchy,
                                                     displayProgress=progressBarPrefix, config=self.__CONFIG,
                                                     verbose=0)
                    del image

                    step = "filtering masks"
                    progressBarPrefix = " - Removing non-sense masks" if not displayOnlyStats else None
                    res = pp.filter_masks(res, bb_threshold=filter_bb_threshold, priority_table=priority_table,
                                          mask_threshold=filter_mask_threshold, verbose=0,
                                          displayProgress=progressBarPrefix, config=self.__CONFIG)

                    if not self.__CORTEX_MODE and len(self.__CUSTOM_CLASS_NAMES) == 10:
                        # TODO : Build automatically classes_hierarchy

                        classes_hierarchy = {
                            3: {"contains": [4, 5], "keep_if_no_child": False},
                            8: {"contains": [9, 10], "keep_if_no_child": True}
                        }
                        step = "filtering orphan masks (pass 2)"
                        progressBarPrefix = " - Removing orphan masks" if not displayOnlyStats else None
                        res = pp.filter_orphan_masks(res, bb_threshold=filter_bb_threshold,
                                                     mask_threshold=filter_mask_threshold,
                                                     classes_hierarchy=classes_hierarchy,
                                                     displayProgress=progressBarPrefix, config=self.__CONFIG,
                                                     verbose=0)

                    if not self.__CORTEX_MODE and len(self.__CUSTOM_CLASS_NAMES) == 10:
                        step = "fusing classes"
                        progressBarPrefix = " - Fusing overlapping equivalent masks" if not displayOnlyStats else None
                        classes_compatibility = [[4, 5]]  # Nsg partiel + nsg complet
                        res = pp.fuse_class(res, bb_threshold=fusion_bb_threshold,
                                            mask_threshold=fusion_mask_threshold,
                                            classes_compatibility=classes_compatibility, config=self.__CONFIG,
                                            displayProgress=progressBarPrefix, verbose=0)

                        step = "removing small masks"
                        progressBarPrefix = " - Removing small masks" if not displayOnlyStats else None
                        res = pp.filter_small_masks(res, min_size=minMaskArea, config=self.__CONFIG,
                                                    displayProgress=progressBarPrefix, verbose=0)

                if not self.__CORTEX_MODE:
                    step = "computing statistics"
                    print(" - Computing statistics on predictions")
                    stats = pp.getCountAndArea(res, classes=self.__CUSTOM_CLASS_NAMES, config=self.__CONFIG)
                    for className in stats:
                        stat = stats[className]
                        print(f"    - {className} : count = {stat['count']}, area = {stat['area']} px")
                    if save_results:
                        with open(os.path.join(image_results_path, f"{imageInfo['NAME']}_stats.json"),
                                  "w") as saveFile:
                            try:
                                json.dump(stats, saveFile, indent='\t')
                            except TypeError:
                                print("    Failed to save statistics", flush=True)

                if save_results:
                    if self.__CORTEX_MODE:
                        step = "cleaning full resolution image"
                        if not displayOnlyStats:
                            print(" - Cleaning full resolution image and saving statistics")
                        allCortices = None
                        # Gathering every cortex masks into one
                        for idxMask, classMask in enumerate(res['class_ids']):
                            if classMask == 1:
                                if allCortices is None:  # First mask found
                                    allCortices = res['masks'][:, :, idxMask].copy() * 255
                                else:  # Additional masks found
                                    allCortices = cv2.bitwise_or(allCortices, res['masks'][:, :, idxMask] * 255)

                        # To avoid cleaning an image without cortex
                        if allCortices is not None:
                            # Extracting the new Bbox
                            allCorticesROI = utils.extract_bboxes(allCortices)

                            # Computing coordinates at full resolution
                            yRatio = imageInfo['HEIGHT'] / self.__CORTEX_SIZE[0]
                            xRatio = imageInfo['WIDTH'] / self.__CORTEX_SIZE[1]
                            allCorticesROI[0] = int(allCorticesROI[0] * yRatio)
                            allCorticesROI[1] = int(allCorticesROI[1] * xRatio)
                            allCorticesROI[2] = int(allCorticesROI[2] * yRatio)
                            allCorticesROI[3] = int(allCorticesROI[3] * xRatio)

                            # Resizing and adding the 2 missing channels of the cortices mask
                            allCortices = cv2.resize(
                                np.uint8(allCortices), (imageInfo['WIDTH'], imageInfo['HEIGHT']),
                                interpolation=cv2.INTER_CUBIC
                            )
                            stats = {"cortex": {"count": 1, "area": dD.getBWCount(allCortices)[1]}}
                            with open(os.path.join(image_results_path, f"{imageInfo['NAME']}_stats.json"),
                                      "w") as saveFile:
                                try:
                                    json.dump(stats, saveFile, indent='\t')
                                except TypeError:
                                    print("    Failed to save statistics", flush=True)

                            temp = np.repeat(allCortices[:, :, np.newaxis], 3, axis=2)

                            # Masking the image and saving it
                            imageInfo['FULL_RES_IMAGE'] = cv2.bitwise_and(
                                imageInfo['FULL_RES_IMAGE'][allCorticesROI[0]: allCorticesROI[2],
                                                            allCorticesROI[1]:allCorticesROI[3], :],
                                temp[allCorticesROI[0]: allCorticesROI[2], allCorticesROI[1]:allCorticesROI[3], :]
                            )
                            cv2.imwrite(os.path.join(image_results_path, f"{imageInfo['NAME']}_cleaned.jpg"),
                                        cv2.cvtColor(imageInfo['FULL_RES_IMAGE'], cv2.COLOR_RGB2BGR),
                                        CV2_IMWRITE_PARAM)

                    if not displayOnlyStats:
                        print(" - Applying masks on image")
                    step = "saving predicted image"
                    fileName = os.path.join(image_results_path, f"{imageInfo['NAME']}_Predicted")
                    # No need of reloading or passing copy of image as it is the final drawing
                    _ = visualize.display_instances(
                        fullImage, res['rois'], res['masks'], res['class_ids'], visualizeNames, res['scores'],
                        colorPerClass=True, fileName=fileName, onlyImage=True, silent=True, figsize=(
                            (1024 if self.__CORTEX_MODE else imageInfo["WIDTH"]) / 100,
                            (1024 if self.__CORTEX_MODE else imageInfo["HEIGHT"]) / 100
                        ), image_format=imageInfo['IMAGE_FORMAT'], config=self.__CONFIG
                    )

                final_time = round(time() - start_time)
                print(f" Done in {formatTime(final_time)}\n")
                step = "finalizing"
                if save_results:
                    with open(logsPath, 'a') as results_log:
                        results_log.write(f"{imageInfo['NAME']}; {final_time};\n")
                del res, imageInfo, fullImage
                plt.clf()
                plt.close('all')
                gc.collect()
            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                failedImages.append(os.path.basename(IMAGE_PATH))
                print(f"/!\\ Failed {IMAGE_PATH} at \"{step}\"\n")
                if save_results and step not in ["image preparation", "finalizing"]:
                    final_time = round(time() - start_time)
                    with open(logsPath, 'a') as results_log:
                        results_log.write(f"{imageInfo['NAME']}; {final_time};FAILED ({step});\n")
        # Saving failed images list if not empty
        if len(failedImages) > 0:
            try:
                with open(os.path.join(results_path, "failed.json"), 'w') as failedJsonFile:
                    json.dump(failedImages, failedJsonFile, indent="\t")
            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                print("Failed to save failed image(s) list. Following is the list itself :")
                print(failedImages)

        total_time = round(time() - total_start_time)
        print(f"All inferences done in {formatTime(total_time)}")
        if save_results:
            with open(logsPath, 'a') as results_log:
                results_log.write(f"GLOBAL; {total_time};\n")

    def save_debug_image(self, step, debugIterator, fullImage, imageInfo, res, image_results_path, names, silent=True):
        if not silent:
            print(f" - Saving {step} image")
        step = step.replace(' ', '_').replace('(', '').replace(')', '')
        fileName = os.path.join(image_results_path, f"{imageInfo['NAME']}_Inference_debug_{debugIterator:02d}_{step}")
        visualize.display_instances(fullImage if self.__LOW_MEMORY else fullImage.copy(), res['rois'], res['masks'],
                                    res['class_ids'], names, res['scores'], colorPerClass=True, fileName=fileName,
                                    onlyImage=False, silent=True, figsize=(
                (1024 if self.__CORTEX_MODE else imageInfo["WIDTH"]) / 100,
                (1024 if self.__CORTEX_MODE else imageInfo["HEIGHT"]) / 100
            ), image_format=imageInfo['IMAGE_FORMAT'], config=self.__CONFIG)
        if self.__LOW_MEMORY:
            del fullImage
            gc.collect()
            fullImage = cv2.cvtColor(cv2.imread(imageInfo['PATH']), cv2.COLOR_BGR2RGB)
