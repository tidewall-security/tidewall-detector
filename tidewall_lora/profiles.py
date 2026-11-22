"""Explicit supported profiles; the preserved detector remains the default."""
from .classifier import Classifier, identity
from .ensemble import EnsembleClassifier, identity as ensemble_identity
from .schemas import PROFILE, ENSEMBLE_PROFILE

PROFILES={
    PROFILE: {'loader':Classifier,'identity':identity,'score_kind':'two_class_softmax_uncalibrated',
              'devices':{'cpu','mps','cuda'}},
    ENSEMBLE_PROFILE: {'loader':EnsembleClassifier,'identity':ensemble_identity,
              'score_kind':'mean_of_two_class_softmax_uncalibrated','devices':{'cpu','metal'}},
}
