# -*- coding: utf-8 -*-
"""
Created on Feb 21 2021
Last modified on Apr 2 2025

@author: Jad Salem, Swati Gupta, Vijay Kamble

This file contains a function which takes in
simulation data and returns density bands
for a fan chart.
"""

import numpy as np


def FanChart(data, band_sizes):
    num_bands = len(band_sizes)
    runs = len(data)
    average = np.mean(data, axis=0)
    low = np.zeros((len(band_sizes),len(average)))
    high = np.zeros((len(band_sizes),len(average)))
    for i in range(len(average)):
        num_less = 0
        for r in range(runs):
            if data[r,i] < average[i]:
                num_less = num_less + 1
        num_greater = runs - num_less 
        if num_less > 0 and num_greater > 0:
            less = np.zeros(num_less)
            greater = np.zeros(num_greater)
            index_less = 0
            index_greater = 0
            for r in range(runs):
                if data[r,i] < average[i]:
                    less[index_less] = -data[r,i]
                    index_less = index_less + 1
                else:
                    greater[index_greater] = data[r,i]
                    index_greater = index_greater + 1
            less = np.sort(less)
            greater = np.sort(greater)
            for b in range(len(band_sizes)):
                low[b,i] = -less[int(np.round(band_sizes[b]*num_less))]
                high[b,i] = greater[int(np.round(band_sizes[b]*num_greater))]
        elif num_less > 0:
            for b in range(len(band_sizes)):
                high[b,i] = average[i]
            less = np.zeros(num_less)
            index_less = 0
            for r in range(runs):
                less[index_less] = -data[r,i]
                index_less = index_less + 1
            less = np.sort(less)
            for b in range(len(band_sizes)):
                low[b,i] = -less[int(np.round(band_sizes[b]*num_less))]
        elif num_greater > 0:
            for b in range(len(band_sizes)):
                low[b,i] = average[i]
            greater = np.zeros(num_greater)
            index_greater = 0
            for r in range(runs):
                greater[index_greater] = data[r,i]
                index_greater = index_greater + 1
            greater = np.sort(greater)
            for b in range(len(band_sizes)):
                high[b,i] = greater[int(np.round(band_sizes[b]*num_less))]
        else:
            for b in range(len(band_sizes)):
                low[b,i] = average[i]
                high[b,i] = average[i]

    return average, low, high