#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 10:24:02 2021

@author: zahid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

#%% Triplet loss

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()
    
class TripletLoss1(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss1, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = distance_positive + F.relu(- distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()
    
    
    
    
class InfoNceLoss(nn.Module):
    def __init__(self, t = 0.05):
        super(InfoNceLoss, self).__init__()
        self.t = t
        self.cos = nn.CosineSimilarity()

    def forward(self, q, k, queue):
    
    
        pos = torch.exp(torch.div(self.cos(q,k),self.t))
        neg = torch.sum(torch.div(torch.exp(self.cos(q, queue)),self.t))
        denominator = neg + pos
    
        return torch.mean(-torch.log(torch.div(pos,denominator)))