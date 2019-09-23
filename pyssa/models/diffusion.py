# abstract model class with interface for diffusion models
from abc import ABC, abstractmethod


class Diffusion(ABC):

    @abstractmethod
    def eval(self, state, time):
        """
        Return a tuple (a(state, time), b(state, time)) where a and b are the drift and diffusion terms respectively
        """
        pass

