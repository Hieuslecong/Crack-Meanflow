"""Compatibility import. New code should use crackmeanflow.conference.losses.meanflow_loss."""
from crackmeanflow.conference.losses.meanflow_loss import ConferenceMeanFlowLoss
CrackMeanFlowLoss = ConferenceMeanFlowLoss
__all__=['CrackMeanFlowLoss','ConferenceMeanFlowLoss']
