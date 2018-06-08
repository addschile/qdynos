from __future__ import absolute_import
import numpy as np
from .utils import dag,norm

class Integrator(object):

    def __init__(self, dt, eom, options):
        """
        """
        self.eom = eom
        self.dt = dt
        self.integrate = None
        self.integrate_range = None
        self.a = None
        self.b = None
        if options.method == 'exact':
            self.integrate = self.exact_integrate
        elif options.method == 'rk4':
            self.order = 4
            self.a = [1./6, 1./3. , 1./3. , 1./6.]
            self.b = [0.0 , 0.5 , 0.5 , 1.]
            self.dt = dt
            self.integrate = self.rk4_integrate
            self.integrate_range = self.rk4_integrate_range
        #elif options.method == 'adams':
        #    self.order = 4
        #    self.a = [1./6, 1./3. , 1./3. , 1./6.]
        #    self.b = [0.0 , 0.5 , 0.5 , 1.]
        #    self.dt = dt
        #    self.integrate = rk4_integrate
        #    self.integrate_range = rk4_integrate_range
        #elif options.method == 'bdf':
        #    self.order = 4
        #    self.a = [1./6, 1./3. , 1./3. , 1./6.]
        #    self.b = [0.0 , 0.5 , 0.5 , 1.]
        #    self.dt = dt
        #    self.integrate = rk4_integrate
        #    self.integrate_range = rk4_integrate_range

    def _set_y_value(self, y, t):
        """
        Sets the function at a certain time. Used for setting the initial
        condition or in restarting a simulation from a restart file.
        """
        self.y = y.copy()
        self.t = t
        return

    def exact_integrate(self):
        """
        This integration will directly update the function based on an exact
        solution of the equation of motion.
        """
        self.y = self.eom(self.y)
        self.t += self.dt

    def rk4_integrate(self, renorm=0):
        """
        This integration uses 4-th order Runge-Kutta to update the function
        based on the equation of motion.
        """
        k = np.zeros_like(self.y)
        dy = np.zeros_like(self.y)

        for i in range(self.order):
            k = self.dt*self.eom( self.b[i]*k + self.y , i)
            dy += self.a[i]*k

        self.y += dy

        if renorm:
            self.y /= np.sqrt(norm(self.y))

        self.t += self.dt

    def rk4_integrate_range(self, t0, t, renorm=1, change_dt=0):
        """
        This integration uses 4-th order Runge-Kutta to update the function
        based on the equation of motion, but does it over a timestep that is
        different the original timestep.

        Used for stochastic unraveling.        
        """
        dt = t-t0
        k = np.zeros_like(self.y)
        dy = np.zeros_like(self.y)

        for i in range(self.order):
            k = dt*self.deriv((self.b[i]*k + self.y), self.deriv_args)
            dy += self.a[i]*k.copy()

        self.y += dy

        if renorm:
            self.y /= np.sqrt(norm(self.y))

        if change_dt:
            self.t += dt
