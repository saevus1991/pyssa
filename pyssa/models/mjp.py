# abstract model class fixing the underlying interface
from abc import ABC, abstractmethod


class MJP(ABC):

    @abstractmethod
    def label2state(self, label):
        """
        Convert a label to the actual state
        """
        pass

    @abstractmethod
    def state2label(self, state):
        """
        Convert a state to the associated label
        """
        pass

    @abstractmethod
    def exit_stats(self, label):
        """
        Returns the exit rate corresponding to the current state
        and an array containing a probability distribution over target states
        """
        pass

    @abstractmethod
    def update(self, state, label):
        """
        Returns the exit rate corresponding to the current state
        and an array containing a probability distribution over target states
        """
        pass    

    def get_event(self, event_ind):
        return(event_ind)

    @property
    def dim(self):
        raise NotImplementedError
