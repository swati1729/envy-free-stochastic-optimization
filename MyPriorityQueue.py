# -*- coding: utf-8 -*-
"""
Created on Jan 28 2022
Last modified on Apr 2 2025

@author: Jad Salem, Swati Gupta, Vijay Kamble

This file creates a priority queue-like class
which is used in the SCAda-LGD implementation
(fig6.py). This code is based on the implemen-
tation found here:
https://www.geeksforgeeks.org/priority-queue-in-python/
"""

import numpy as np

class MyPriorityQueue(object):
    def __init__(self):
        self.queue = []
    def __str__(self):
        return "\n".join(["\n".join(["{0}: {1}".format(key, pair[key]) for key in pair]) for pair in self.queue])
    # checking if queue is empty
    def isEmpty(self):
        return len(self.queue) == 0
    def hasFeasiblePoints(self, arg):
        if self.isEmpty():
            return False
        max_index = None
        max_int = None
        max_key = None
        for i in range(len(self.queue)):
            pair_key = list(self.queue[i].keys())[0]
            pair_int = self.queue[i][pair_key]
            if (max_index == None) or (pair_int < max_int):
                max_index = i
                max_int = pair_int
                max_key = pair_key
        if max_int <= arg:
            return True
        else:
            return False
    # insert an element
    def insert(self, data):
        if (type(data) == dict) and (len(data) == 1):
            self.queue.append(data)
    
    # pop an element
    def delete(self):
        if self.isEmpty():
            return [None, None]
        max_index = None
        max_int = None
        max_key = None
        for i in range(len(self.queue)):
            pair_key = list(self.queue[i].keys())[0]
            pair_int = self.queue[i][pair_key]
            if (max_index == None) or (pair_int < max_int):
                max_index = i
                max_int = pair_int
                max_key = pair_key
        del self.queue[max_index]
        return [max_key, max_int]
    