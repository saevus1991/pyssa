# kinetic model class storing the pre, post and adjecency matrix of a kinetic model


class KineticModel:

    # constructor
    def __init__(self, pre, post, rates):
        self.pre = pre
        self.post = post
        self.rates = rates
        self.stoichiometry = post-pre
