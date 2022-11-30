# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 18:14:40 2022

@author: xdzl45
"""

####################################################################################################
#
#   ERROR CLASS FOR MALFORMED ESTIMATOR ARGUMENTS
#
####################################################################################################

class ExpressionError(Exception):
    r"""
    Exception class for malformed expressions in the input
    """
    def __init__(self, expression, msg):
        self.expression = expression
        self.msg = msg
        
    def __str__(self):
        return "[%s] %s" % (self.expression, self.msg)


####################################################################################################
#
#   WARNING CLASS FOR PREMATURELY TERMINATED SCF ITERATIONS
#
####################################################################################################

class NotConvergedWarning(Exception):
    r"""
    Exception class for non-convergence of estimators
    """
    def __init__(self, estimator, increment):
        self.estimator = estimator
        self.increment = increment
    def __str__(self):
        return "[%s] only reached increment %.3e" % (self.estimator, self.increment)