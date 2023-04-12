import numpy as np
import matplotlib.pyplot as plt
import functools 
import sympy as sp
import time
import autograd as grd
import math

class newton:
	def __init__(self, tol):
		self.tol = tol

	def __call__(self, f, x):
		g = grd.grad(f)
		err = math.inf
		
		while err >= self.tol:
			x, y = x - f(x)/g(x), x
			err = abs(x - y)/y

		return x

class adams:
	def __call__(self, f, a, b, alpha, h):
		t_axis = np.arange(a, b, h)
		
		assert(len(t_axis) > self.sstep)
		
		y_axis = np.empty(len(t_axis)); y_axis[0] = alpha
		
		for i, t in enumerate(t_axis[1:self.sstep], 1):
			y_axis[i] = y_axis[i-1] + h*f(t, y_axis[i-1])
		
		return t_axis, y_axis
		
class adams_bashforth(adams):
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
			y_axis[i] = y_axis[i-1] + h*(steps@self.coeffs)
			steps[1:] = steps[:self.sstep-1]
			steps[0] = f(t_axis[i], y_axis[i])

		return t_axis, y_axis

class adams_moulton(adams):
	def __init__(self, sstep, tol):
		self.sstep = sstep
		self.coeffs = self.__get_coeffs(sstep)
		self.tol = tol
		self.implicit_solver = newton(tol)


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
			def find_my_root(w): return w - y_axis[i-1] - h*(f(t, w)*self.coeffs[0] + steps@self.coeffs[1:])

			y_axis[i] = self.implicit_solver(find_my_root, y_axis[i-1])

			steps[1:] = steps[:self.sstep-1]
			steps[0] = f(t_axis[i], y_axis[i])

		return t_axis, y_axis


my_integrator = adams_moulton(12, 1e-9)

t, y = my_integrator(lambda t,y: -t*y, -4, 4, .02, .008)


real_y = 59.61 * np.exp(-t**2)
plt.plot(t, y)
plt.plot(t, real_y)
plt.show()



