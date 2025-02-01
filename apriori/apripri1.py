#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 23:12:39 2025

@author: efekanbas
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv("sepet.csv", header = None)

t = []
for i in range (0,7501):
    t.append([str(veriler.values[i,j]) for j in range (0,20)])


from apyori import apriori

kurallar = apriori(t,min_support = 0.01,min_confidence = 0.2,min_lift = 3,min_length=2)

print(list(kurallar))