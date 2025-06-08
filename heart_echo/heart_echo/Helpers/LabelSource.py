import pandas as pd
import os
import pathlib
import numpy as np
from .Helpers import Helpers
from enum import Enum


class LABELTYPE(Enum):
    NONE = 0
    VISIBLE_FAILURE_WEAK = 1
    VISIBLE_FAILURE_STRONG = 2
    AGE = "Age"
    MATURITY_AT_BIRTH = "Maturity at birth"
    PRETERM_BINARY = 3
    PRETERM_CATEGORICAL = 4
    VIEW = 5
    PRETERM_CATEGORICAL_ALTERNATE = 6
    ID = 7
    HEART_FAILURE = "HF"
    PULMONARY_HYPERTENSION = "PH"  # NOTE: edited (fixed typo)


# Preterm definitions from https://www.who.int/news-room/fact-sheets/detail/preterm-birth
class Preterm:
    class Categories(Enum):
        EXTREME = 0
        VERY = 1
        MODERATE = 2
        NOT_PRETERM = 3

    Values = {Categories.EXTREME: 195,  # Less than 28 weeks
              Categories.VERY: 223,  # 28 to 32 weeks
              Categories.MODERATE: 258,  # Less than 37 weeks
              Categories.NOT_PRETERM: 259  # 37 weeks or more
              }

    @staticmethod
    def one_hot(category):
        ret_arr = np.zeros((len(Preterm.Categories)), dtype=int)
        ret_arr[category.value] = 1
        return ret_arr


class PretermAlternate:
    class Categories(Enum):
        VERY = 0
        MODERATE = 1
        NOT_PRETERM = 2

    Values = {
        Categories.VERY: 193,
        Categories.MODERATE: 258,
        Categories.NOT_PRETERM: 259
    }

    @staticmethod
    def one_hot(category):
        ret_arr = np.zeros((len(Preterm.Categories)), dtype=int)
        ret_arr[category.value] = 1
        return ret_arr


class LabelSource:
    def __init__(self, test):
        
        path = None
        if test == False: path = "video_angles_list.csv"
        else: path = "video_angles_list_test.csv"

        self._table = pd.read_csv(os.path.join(pathlib.Path(__file__).parent.absolute(), path),
                                  usecols=["Patient", "Age", "Maturity at birth", "HF", "PH"] + Helpers.ALL_VIDEO_ANGLES,
                                  index_col="Patient")

    def get_label_for_patient(self, patient_ID, label_type):
        if label_type in [LABELTYPE.VISIBLE_FAILURE_WEAK, LABELTYPE.VISIBLE_FAILURE_STRONG]:
            raise NotImplementedError("Only Age or Maturity at birth currently supported")

        elif label_type is LABELTYPE.NONE:
            return 0

        elif label_type is LABELTYPE.PRETERM_BINARY:
            gestation_days = self._table.loc[patient_ID][LABELTYPE.MATURITY_AT_BIRTH.value]
            return 0 if gestation_days >= Preterm.Values[Preterm.Categories.NOT_PRETERM] else 1

        elif label_type is LABELTYPE.PRETERM_CATEGORICAL:
            gestation_days = self._table.loc[patient_ID][LABELTYPE.MATURITY_AT_BIRTH.value]

            if gestation_days <= Preterm.Values[Preterm.Categories.EXTREME]:
                category = Preterm.Categories.EXTREME
            elif gestation_days <= Preterm.Values[Preterm.Categories.VERY]:
                category = Preterm.Categories.VERY
            elif gestation_days <= Preterm.Values[Preterm.Categories.MODERATE]:
                category = Preterm.Categories.MODERATE
            else:
                category = Preterm.Categories.NOT_PRETERM

            return Preterm.one_hot(category)

        elif label_type is LABELTYPE.PRETERM_CATEGORICAL_ALTERNATE:
            gestation_days = self._table.loc[patient_ID][LABELTYPE.MATURITY_AT_BIRTH.value]

            if gestation_days <= PretermAlternate.Values[PretermAlternate.Categories.VERY]:
                category = PretermAlternate.Categories.VERY
            elif gestation_days <= PretermAlternate.Values[PretermAlternate.Categories.MODERATE]:
                category = PretermAlternate.Categories.MODERATE
            else:
                category = PretermAlternate.Categories.NOT_PRETERM

            return PretermAlternate.one_hot(category)

        elif label_type is LABELTYPE.ID:
            return patient_ID

        else:
            return self._table.loc[patient_ID][label_type.value]

    def get_label_for_patient_and_view(self, patient_ID, view, label_type):
        if label_type not in [LABELTYPE.VISIBLE_FAILURE_WEAK, LABELTYPE.VISIBLE_FAILURE_STRONG]:
            raise NotImplementedError("Only VISIBLE_FAILURE_WEAK or VISIBLE_FAILURE_WEAK supported")

        if view not in Helpers.ALL_VIDEO_ANGLES:
            raise ValueError("Invalid input {}".format(view))

        if label_type == LABELTYPE.VISIBLE_FAILURE_WEAK:
            if int(self._table.loc[patient_ID][view]) in [1, 2]:
                return 1
            else:
                return 0

        else:
            if (self._table.loc[patient_ID][view]) == 2:
                return 1
            else:
                return 0
