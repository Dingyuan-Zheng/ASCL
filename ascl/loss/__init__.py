from __future__ import absolute_import

from .triplet import TripletLoss, SoftTripletLoss
from .crossentropy import CrossEntropyLabelSmooth, SoftEntropy
from .loss import AALS, PGLR, InterCamProxy
from .adasp_loss import AdaSPLoss
from .maximum_mean_discrepancy import MaximumMeanDiscrepancy

__all__ = [
    'TripletLoss',
    'CrossEntropyLabelSmooth',
    'SoftTripletLoss',
    'SoftEntropy',
    'AALS',
    'PGLR',
    'InterCamProxy',
    'AdaSPLoss',
    'MaximumMeanDiscrepancy'
]