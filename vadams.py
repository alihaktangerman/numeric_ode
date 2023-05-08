import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import functools
from scipy import optimize

"""
TODO:
2-add variable step-size support
3-
"""

class vector_euler:
	def __call__(self, f, a, b, alpha, h):
		t_space = np.arange(a, b, h)
		y_space = np.empty([len(t_space), len(alpha)])
		y_space[0] = alpha
		for i, t in enumerate(t_space[1:], 1):
			y_space[i] = y_space[i-1] + h*f(t, y_space[i-1])
		return t_space, y_space

class vector_adams:
	def __call__(self, f, a, b, alpha, h):
		t_axis = np.arange(a, b, h)
		
		assert(len(t_axis) > self.sstep)
		
		y_axis = np.empty([len(t_axis), len(alpha)]);
		y_axis[0] = alpha
		
		for i, t in enumerate(t_axis[1:self.sstep], 1):
			y_axis[i] = y_axis[i-1] + h*f(t, y_axis[i-1]) #modify to be compatible with other single step methods
		
		return t_axis, y_axis

class vector_adams_bashforth(vector_adams):
	def __init__(self, sstep):
		self.sstep = sstep
		self.coeffs = self.__get_coeffs(sstep)

	@functools.lru_cache
	def __get_coeffs(self, sstep):
		u = sp.Symbol('u')
		#https://en.wikipedia.org/wiki/Linear_multistep_method
		ret = np.array([sp.integrate(sp.prod(u+i if i != j else 1 for i in range(sstep)), (u, 0, 1))
						* (-1)**j / (sp.factorial(j)*sp.factorial(sstep-j-1)) for j in range(sstep)], dtype=float)

		return ret

	def __call__(self, f, a, b, alpha, h):
		t_axis, y_axis = super().__call__(f, a, b, alpha, h)
		
		steps = np.array([f(t_axis[i], y_axis[i]) for i in range(self.sstep-1, -1, -1)])
		
		for i, t in enumerate(t_axis[self.sstep:], self.sstep):
			y_axis[i] = y_axis[i-1] + h*sum(s*c for s,c in zip(steps,self.coeffs))
			steps[1:] = steps[:self.sstep-1]
			steps[0] = f(t_axis[i], y_axis[i])

		return t_axis, y_axis

class vector_adams_moulton(vector_adams):
	def __init__(self, sstep):
		self.sstep = sstep
		self.coeffs = self.__get_coeffs(sstep)

	@functools.lru_cache
	def __get_coeffs(self, sstep):
		u = sp.Symbol('u')
		#https://en.wikipedia.org/wiki/Linear_multistep_method
		ret = np.array([sp.integrate(sp.prod(u+i-1 if i != j else 1 for i in range(sstep+1)), (u, 0, 1))
		 				* (-1)**j / (sp.factorial(j)*sp.factorial(sstep-j)) for j in range(sstep+1)], dtype=float)

		return ret

	def __call__(self, f, a, b, alpha, h):
		t_axis, y_axis = super().__call__(f, a, b, alpha, h)
		
		steps = np.array([f(t_axis[i], y_axis[i]) for i in range(self.sstep-1, -1, -1)])

		for i, t in enumerate(t_axis[self.sstep:], self.sstep):
			def find_my_root(w): 
				return w - y_axis[i-1] - h*(f(t, w)*self.coeffs[0] + sum(c*s for c,s in zip(steps,self.coeffs[1:])))

			y_axis[i] = optimize.root(find_my_root, y_axis[i-1]).x

			steps[1:] = steps[:self.sstep-1]
			steps[0] = f(t_axis[i], y_axis[i])

		return t_axis, y_axis

class vector_adams_bashorth_moulton(vector_adams_moulton, vector_adams_bashforth):
	def __init__(self, sstep, sstep_xpl, tol):
		assert(sstep > sstep_xpl)
		self.sstep = sstep
		self.sstep_xpl = sstep_xpl
		self.coeffs = self._vector_adams_moulton__get_coeffs(sstep)
		self.coeffs_xpl = self._vector_adams_bashforth__get_coeffs(sstep_xpl)
		self.tol = tol

	def __call__(self, f, a, b, alpha, h):
		t_axis, y_axis = vector_adams.__call__(self, f, a, b, alpha, h)

		steps = np.array([f(t_axis[i], y_axis[i]) for i in range(self.sstep-1, -1, -1)])

		for i, t in enumerate(t_axis[self.sstep:], self.sstep):
			y_predicted = y_axis[i-1] + h*(sum(c*s for c,s in zip(self.coeffs_xpl,steps[:self.sstep_xpl])))
			y_corrected = y_axis[i-1] + h*(f(t, y_predicted)*self.coeffs[0] + sum(c*s for c,s in zip(self.coeffs[1:],steps)))
			
			while np.linalg.norm(y_predicted - y_corrected) > self.tol:
				y_predicted = y_corrected
				y_corrected = y_axis[i-1] + h*(f(t, y_predicted)*self.coeffs[0] + sum(c*s for c,s in zip(self.coeffs[1:],steps)))

			y_axis[i] = y_predicted

			steps[1:] = steps[:self.sstep-1]
			steps[0] = f(t_axis[i], y_axis[i])

		return t_axis, y_axis

"""
class variable_stepsize_vector_adams_bashforth_moulton:
	def __init__(self, sstep, sstep_xpl, tol):
		assert(sstep > sstep_xpl):
		self.sstep = sstep
		self.sstep_xpl = sstep_xpl
		self.coeffs = self._vector_adams_moulton__get_coeffs(sstep)
		self.coeffs_xpl = self._vector_adams_bashforth__get_coeffs(sstep_xpl)
		self.tol = tol

	def __call__(self, f, a, b, alpha, h):
		t_axis, y_axis = np.linspace(0, h_min*self.sstep, num=self.sstep), np.empty(self.sstep)
		y_axis[0] = alpha

		for i,t in enumerate(y_axis[1:],1):
			y_axis[i] = y_axis[i-1] + h*f(t,y_axis[i-1])

		start = self.sstep

		while start < b:
			

			check_me = abs(y_predicted - y_corrected)

			if check_me < tol:
"""



def f(t, y, a, b, c, d):
	assert(len(y) == 2)
	return np.array([a*y[0] - b*y[0]*y[1], -c*y[1] + d*y[0]*y[1]])

def h(t, y):
	return f(t, y, 1.5,1,3,1)

def g(t, y):
	return np.array([y[1],-y[0]])
my_integrator = vector_adams_bashorth_moulton(4,3,1e-5)
t, y = my_integrator(h, 0 , 20, np.array([10, 5]), .05)

plt.plot(t, y)
plt.show()

