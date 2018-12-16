# -*- coding: utf-8 -*-
"""
Created: Dec 2010
Script that integrates a set of differential equations using a stoichiometric
matrix (stoi), and incorporates Monod kinetics in the rate equations.
It is used to simulate the dynamics of a metabolic model

This model is based on the simple metabolic system proposed in "A simple 
self-maintaining metabolic system: robustness, autocatalysis, bistability" 
(Piedrafita, Montero et al, 2010, PLOS computational biology, vol 6, issue 8, 
 august 2010)

**Inhibition and activation not functional
**calculations can be simplified into matrix form (?)

@author: naabuijs
"""
import numpy as np
from scipy import integrate
import matplotlib.pylab as plt


class Model(object):
    def __init__(self):
        # simulation details, these variables are used as input for the integration step
        self.startTime = 0      # start of simulation
        self.stopTime = 2500    # end of simulation
        self.steps = 10000      # steps in between startTime and stopTime

        # concentrations external reactants <- Monod constants?
        self._S = 4
        self._T = 2
        self._U = 1

        # initial conditions and time points array
        self._X0 = np.array([20., 0., 0., 0., 0., 0., 0., 0.])
        self._t = np.linspace(self.startTime, self.stopTime, self.steps)

        # attributes for storing integration results
        self._X = np.array([])
        self._info_dict = dict()

    def _dX_dt(self, X, t=0):
        """ this function description mass balances / defines differential equations as a function
        it would be nicer to list all model parameters in the beginning of this script
        and use them as an input for the function dX_dt, but I'm not sure if this
        can be done through the integrate.odeint function
        """
        # stoichiometric matrix, rows = reactions, columns = reactants involved
        stoi = np.array([
                        [-1., 1., 0, 0, 0, 0, 0, 0],
                        [1., -1, 0, 0, 0, 0, 0, 0],
                        [0, -1., 1., 0, 0, 0, 0, 0],
                        [0, 1., -1., 0, 0, 0, 0, 0],
                        [1., 0, -1., 1., 0, 0, 0, 0],
                        [-1., 0, 1., -1., 0, 0, 0, 0],
                        [-1., 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, -1., 1., 0, -1., 0],
                        [0, 0, 0, 1., -1., 0, 1., 0],
                        [0, 0, 0, 0, -1., 1., 0, 0],
                        [0, 0, 0, 0, 1., -1., 0, 0],
                        [1., 0, 0, 0, 0, -1., 1., 0],
                        [-1., 0, 0, 0, 0, 1., -1., 0],
                        [0, 0, 0, 0, 0, 0, -1., 0],
                        [0, -1., 0, 0, 0, 0, 0, 1.],
                        [0, 1., 0, 0, 0, 0, 0, -1.],
                        [1., 0, 0, 0, 0, 0, 1., -1.],
                        [-1., 0, 0, 0, 0, 0, -1., 1.],
                        [0, 0, 0, -1., 0, 0, 0, 0]
                        ])
        # array containing information whether the reaction (rows correspond to
        # reactions as in stoi) is inhibited (=-1) or activated (=1) by a metabolite
        # (columns correspond to metabolite as in stoi)
        inh_act = np.array([
                            [0, 0, 0, 0, 0],
                            [0, 0, -1., 0, 0],
                            [0, 0, 0, 0, -1.],
                            [-1., 0, 0, 0, 0],
                            [-1., 0, 0, -1., 0]
                        ])

        inh_act_const = np.array([
                                    [0, 0, 2., 0, 0],
                                    [0, 0, 2., 0, 0],
                                    [0, 0, 0, 0, 2.],
                                    [1., 0, 0, 0, 0],
                                    [1., 0, 0, 1., 0]
                                ])

        # these two variables only serve the purpose of keeping track in which direction
        # the stoichiometric matrix is being read, meaning column-wise or row-wise.
        no_reactants = stoi.shape[1]            # use this to walk through the arrays later on
        no_reactions = stoi.shape[0]            # same purpose

        rates = np.ones((no_reactions, 1))          # array is filled with correct rates later on
        # maximum rates for each reaction:
        r_max = (10., 10., 10, 10., 2., 1., 0.3, 1., 1., 1., 1., 0.1, 0.1, 0.3, 0.1, 0.05, 0.05, 0.05, 0.3)
        # Monod constant for each reaction
        const = (self._S, 1., self._T, 1., 1., 1., 1., 1., 1., self._U, 1., 1., 1., 1., self._U, 1., 1., 1., 1.)
        inhibition = np.ones((no_reactions, 1))     # one means no inhibition
        activation = np.ones((no_reactions, 1))     # one means no activation
        inh_den = np.zeros((no_reactions, 1))  # array for denominator term in rate equations, zero means no inhibition
        act_den = np.zeros((no_reactions, 1))  # array for denominator term in rate equations, zero means no activation

        # determine inhibition and activation for each reactions
        # 1st walk through each row/reaction
        for i in range(0, inh_act.shape[0]):
            # for each row walk through each column/metabolite to determine
            # whether it inhibits or activates a reaction
            for j in range(0, inh_act.shape[1]):
                # the sign of the value in array inh_act tells whether it inhibits
                # or activates the reaction in question
                if inh_act[i, j] == -1:
                    Ki = inh_act_const[i, j]    # inhibition constant
                    Xi = X[j]                  # concentration inhibiting compound

                    inhibition[i, 0] *= (1 - (Xi/Ki))
                    inh_den[i, 0] -= (Xi/Ki)

                elif inh_act[i, j] == 1:
                    Ka = inh_act_const[i, j]    # activation constant
                    Xa = X[j]                  # concentration activation compound

                    activation[i, 0] *= (1 - (Xa/Ka))
                    act_den[i, 0] += (Xa/Ka)

        # calculate rates based on concentrations in X, and the functions defined above
        # create an array that is filled with the concentration term in the kinetic function rates
        x_for_rates = np.ones((no_reactions, 1))

        for g in range(0, no_reactions):
            for h in range(0, no_reactants):
                if stoi[g, h] == -1:
                    # determine parameters for rate function
                    x_for_rates[g] *= X[h]    # select substrate concentration from given array X
                    con = const[g]           # select constant for this reaction
                    rm = r_max[g]             # select maximum rate for this reaction
                    inh = inhibition[g, 0]    # inhibition term
                    inh_d = inh_den[g, 0]     # inhibition term in denominator
                    act = activation[g, 0]    # activation term
                    act_d = act_den[g, 0]     # activation term in denominator

            # calculate rate using rate function for one substrate reactions
            rates[g, 0] = self._rate(rm, x_for_rates[g,0], con)

        # now combine the multiply the array stoi by rates, inhibition, and activation
        # to determine the change in metabolite levels for each metabolite

        # 1st, define an array that is returned by this function
        change = np.zeros(no_reactants)
        count_i = no_reactants
        count_j = no_reactions

        for k in range(0, count_i):
            for l in range(0, count_j):
                if stoi[l, k] != 0:
                    change[k] += stoi[l, k]*rates[l, 0] #* inhibition[l, 0] * activation[l, 0]
        return change

    def _integrate_model_equations(self):
        """
        integration of diff eq function
        """
        # integrate diff eq function over time span t with initial values X0
        self._X, self._info_dict = integrate.odeint(self._dX_dt, self._X0, self._t, full_output=True)
        print(self._info_dict['message'])

    def _plot_integration_results(self):
        """
        plotting of results
        """
        STU, STUS, STUST, ST, SUST, SUSTU, SU, STUSU = self._X.T
        f1 = plt.figure()
        plt.plot(self._t, STU, 'b-', label='STU')
        plt.plot(self._t, STUS, 'b--', label='STUS')
        plt.plot(self._t, STUST, 'r-', label='STUST')
        plt.plot(self._t, ST, 'g-', label='ST')
        plt.plot(self._t, SUST, 'y-', label='SUST')
        plt.plot(self._t, SUSTU, 'm-', label='SUSTU')
        plt.plot(self._t, SU, 'c-', label='SU')
        plt.plot(self._t, STUSU, 'r--', label='STUSU')
        plt.grid()
        plt.legend(loc='best')
        plt.xlabel('Time')
        plt.ylabel('Concentration')
        plt.title('A simple self-maintaining metabolic system')
        f1.savefig('Piedrafita model output.png')

    @staticmethod
    def _rate(r_max, x, const):
        """
        a Monod function for a one substrate reaction with product inhibition and activation

        :param r_max: reaction rate
        :param x: substrate concentration
        :param const: reaction rate constant
        :return: calculated reaction rate
        """
        rate = r_max * const * x

        return rate

    def integrate(self):
        self._integrate_model_equations()
        self._plot_integration_results()


if __name__ == "__main__":
    model = Model()
    model.integrate()
