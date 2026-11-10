__author__ = 'mhgou'
__version__ = '1.2.11'

from .grasp import Grasp, GraspGroup, RectGrasp, RectGraspGroup


def __getattr__(name):
    if name == 'GraspNet':
        from .graspnet import GraspNet
        return GraspNet
    if name == 'GraspNetEval':
        from .graspnet_eval import GraspNetEval
        return GraspNetEval
    raise AttributeError("module 'graspnetAPI' has no attribute '{}'".format(name))
