#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" `result_names` class property """


class ResNames(type):
    
    ''' Metaclass for class property in models '''

    @property
    def result_names(cls):
        ''' Names of the values returned by the model '''
        return cls._res_names
