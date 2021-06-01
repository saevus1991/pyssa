# kinetic model class storing the pre, post and adjecency matrix of a kinetic model

#imports
import numpy as np
from collections import OrderedDict
import re
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
import sys
from scipy import sparse
import pyssa.util as ut


class KineticModelBuilder():

    def __init__(self, name='Model', volume=1.0):
        self.name = name
        self.num_species = 0
        self.num_reactions = 0
        self.species_dict = OrderedDict()
        self.reaction_dict = OrderedDict()
        self.rates_dict = OrderedDict()
        self.rates = None
        self.pre = None
        self.post = None
        self.stoichiometry = None
        self.is_build = False
        self.volume = volume
        self.volume_factor = None
        return

    def add_species(self, species):
        """
        Ad species to the list of species
        Input
            spcecies: string identifier for the species, should not contain spaces
        """
        self.species_dict[species] = self.num_species
        self.num_species += 1
        return

    def add_reaction(self, name, reaction, rate):
        """
        Ad reaction to reaction dict
        Input
            name: a string identifier for the reaction
            reaction: a string of the form "a A + b B -> c C + d D" 
                      with integers a, b, c, d and valid species A, B, C, D
            rate: float corresponding to the rate
        """
        self.reaction_dict[name] = reaction
        self.rates_dict[name] = rate
        self.num_reactions += 1
        return

    def remove_reaction(self, name):
        """
        Remove reaction to reaction dict
        Input
            name: a string identifier for the reaction
        """
        del self.reaction_dict[name]
        del self.rates_dict[name]
        self.num_reactions -= 1
        return

    def build(self):
        """
        Build model from current species and reactions
        """
        # initialize
        pre = np.zeros((self.num_reactions, self.num_species))
        post = np.zeros((self.num_reactions, self.num_species))
        rates = np.zeros(self.num_reactions)
        # process reactions
        for i, (key, reaction) in enumerate(self.reaction_dict.items()):
            # split reaction in left and right side
            substrates, products = re.split(' -> ', reaction)
            # extract inputs 
            substrates = re.split(' \+ ', substrates)
            for substrate in substrates:
                num, species = re.split(' ', substrate)
                pre[i, self.species_dict[species]] = int(num)
            # extract outputs
            products = re.split(' \+ ', products)
            for product in products:
                num, species = re.split(' ', product)
                post[i, self.species_dict[species]] = int(num)
            # get rates
            rates[i] = self.rates_dict[key]
        self.pre = pre
        self.post = post
        self.rates = rates
        self.stoichiometry = post-pre
        self.is_build = True
        self.volume_factor = self.volume**(np.sum(self.pre, axis=1)-1)
        return

    def reset(self):
        """
        Reset model (delete numpy arrays)
        """
        self.pre = None
        self.post = None
        self.rates = None
        self.is_build = False

    def to_sbml(self, file_path=None, initial=None):
        """
        Convert model to an sbml file and store at file_path
        """
        if file_path is None:
            file_path = sys.argv[0].replace('.py', '.xml')
        # initialize model
        sbml = ET.Element('sbml')
        sbml.set('level', '2')
        sbml.set('version', '5')
        sbml.set('xmlns', 'http://www.sbml.org/sbml/level2/version5')
        model = ET.SubElement(sbml, 'model')
        model.set('name', self.name)
        # set compartments
        compartment_list = ET.SubElement(model, 'listOfCompartments')
        compartment = ET.SubElement(compartment_list, 'compartment')
        compartment.set('id', 'cell')
        compartment.set('size', '1e-15')
        # set up species
        species_list = ET.SubElement(model, 'listOfSpecies')
        for i, (key, val) in enumerate(self.species_dict.items()):
            species = ET.SubElement(species_list, 'species')
            species.set('id', f'X_{i}')
            species.set('name', key)
            species.set('compartment', 'cell')
            if initial is not None:
                species.set('initialAmount', str(initial[i]))
        # set up reaction list
        reaction_list = ET.SubElement(model, 'listOfReactions')
        for i, (key, val) in enumerate(self.reaction_dict.items()):
            reaction = ET.SubElement(reaction_list, 'reaction')
            reaction.set('id', f'R_{i}')
            reaction.set('name', key)
            reaction.set('reversible', 'false')
            reactant_list = ET.SubElement(reaction, 'listOfReactants')
            product_list = ET.SubElement(reaction, 'listOfProducts')
            # define kinetic law
            kinetic_law = ET.SubElement(reaction, 'kineticLaw')
            math = ET.SubElement(kinetic_law, 'math')
            math.set('xmlns', 'http://www.w3.org/1998/Math/MathML')
            apply = ET.SubElement(math, 'apply')
            ET.SubElement(apply, 'times')
            ci = ET.SubElement(apply, 'ci')
            ci.text = f'c_{i}'
            parameter_list = ET.SubElement(kinetic_law, 'listOfParameters')
            parameter = ET.SubElement(parameter_list, 'parameter')
            parameter.set('id', f'c_{i}')
            parameter.set('value', str(self.rates_dict[key]))
            # get reactants and products
            substrates, products = re.split(' -> ', val)
            substrates = re.split(' \+ ', substrates)
            products = re.split(' \+ ', products)
            # construct reactant list
            for substrate in substrates:
                species_ref = ET.SubElement(reactant_list, 'speciesReference')
                num, species = re.split(' ', substrate)
                species = f'X_{self.species_dict[species]}'
                species_ref.set('species', species)
                species_ref.set('stoichiometry', num)
                ci = ET.SubElement(apply, 'ci')
                ci.text = species
            # construct product list
            for product in products:
                species_ref = ET.SubElement(product_list, 'speciesReference')
                num, species = re.split(' ', product)
                if int(num) > 0:
                    species = f'X_{self.species_dict[species]}'
                    species_ref.set('species', species)
                    species_ref.set('stoichiometry', num)
                else:
                    # introduce sink
                    sink = ET.SubElement(species_list, 'species')
                    sink_id = f'X_{self.species_dict[species]}_'
                    sink.set('id', sink_id)
                    sink.set('sboTerm', 'SBO:0000291')
                    sink.set('compartment', 'cell')
                    # set as reference
                    species_ref.set('species', sink_id)
            # 
            # times = ET.SubElement(apply, 'times')
            # ci = ET.SubElement(apply, 'ci')
        # parse output 
        xmlstr = minidom.parseString(ET.tostring(sbml)).toprettyxml(indent='    ', encoding='utf-8')  # .encode('utf-8')
        with open(file_path, "wb") as f:
            f.write(xmlstr)
        return


class KineticModel(KineticModelBuilder):
    """
    Implementation of a basic mass-action gillespie model 
    """

    # implemented abstract methods

    def __init__(self, name='Model', pre=None, post=None, rates=None, volume=1.0):
        """
        The class requires 2 matrices of shape (num_reactions x num_species)
        specifying the populations before and after each reaction as well
        as an array of length num_reactions specifying the time scale of each reaction
        """
        KineticModelBuilder.__init__(self, name, volume)
        if pre is not None and post is not None and rates is not None:
            self.num_reactions, self.num_species = pre.shape
            self.pre = pre
            self.post = post
            self.rates = rates
            self.stoichiometry = post-pre
            self.volume_factor = self.volume**(np.sum(self.pre, axis=1)-1)

    def label2state(self, label):
        """
        For a kinetic model, this works on the level of reactions,
        i.e a reaction index is mapped to a state change
        """
        return(label)

    def state2label(self, state):
        """
        For a kinetic model, state and label
        """
        return(state)

    def propensity(self, state):
        prop = self.mass_action(state) * self.rates / self.volume_factor
        return(prop)

    def exit_stats(self, state):
        """
        Returns the exit rate corresponding to the current state
        and an array containing a probability distribution over target states
        """
        # compute raw mass action propensity
        prop = self.propensity(state)
        # compute rate and
        rate = prop.sum()
        # catch for absorbing states
        if rate == 0.0:
            transition = np.zeros(prop.shape)
        else:
            transition = prop/rate
        return(rate, transition)

    def update(self, state, event):
        """
        Update the state using the current reaction index
        """
        new_state = state+self.stoichiometry[event]
        return(new_state)

    # additional functions

    # def event2change(self, label):
    #     """
    #     For a kinetic model, this works on the level of reactions,
    #     i.e a reaction index is mapped to a state change
    #     """
    #     return(self.stoichiometry[label, :])

    def mass_action(self, state):
        """
        Compute the mass-action propensity
        """
        # initialize with ones
        prop = np.ones(self.num_reactions)
        # iterate over reactions
        for i in range(self.num_reactions):
            for j in range(self.num_species):
                prop[i] *= ut.falling_factorial(state[j], self.pre[i, j])
        return(prop)

