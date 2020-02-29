"""Module to peform simulation analysis"""

import numpy
import skimage.measure

class bubbleprops(object):

    """
    Class to store and compute bubble properites using the level-set function
    """

    def __init__(self,data):

       """
       Arguments
       ---------

       data : object
            SimulationData object which contains information of computed variables

       """

       self._get_bubbleprops(data)


    def _get_bubbleprops(data):
        self.bubbleprops # = function(data.fields['dfun'][tindex,blockindex,:,:,:])
