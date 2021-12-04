#!/usr/bin/python3

from which_pyqt import PYQT_VER
# from PriorityQueueHeap import PriorityQueueHeap as hpq
from RandomArray import RandomArray as hpq
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import copy
import numpy as np
import random as rand
from TSPClasses import *
import heapq
import itertools



class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''
	
	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	def greedy( self,time_allowance=60.0 ):
		# get prepared
		cities = self._scenario.getCities()
		num_cities = len(cities)
		best_min = math.inf
		best_route = []
		count = 0
		start_time = time.time()
		# iterate through all cities for starting points
		for start_city in cities:
			# keep track of visited cities
			visited = [start_city]
			next = start_city
			curr_length = 0
			# iterate through other cites till all cities visited
			while True:
				min = math.inf
				min_city = None
				# find next shortest route
				for child in cities:
					if child in visited:
						continue
					edge = next.costTo(child)
					if edge < min:
						min = edge
						min_city = child
				# no solution exists
				if min == math.inf:
					break
				curr_length += min
				next = min_city
				visited.append(min_city)
				# we've made a circuit
				if next == start_city:
					break
			# We've found a better solution
			if next.costTo(start_city) != math.inf and len(visited) == num_cities:
				count += 1
				if curr_length < best_min:
					best_route = visited
					best_min = curr_length
			if time.time() - start_time > time_allowance:
				break
		bssf = TSPSolution(best_route)
		results = {}
		results["cost"] = bssf.cost
		results["time"] = time.time() - start_time
		results["count"] = count
		results["soln"] = bssf
		results["max"] = None
		results["total"] = None
		results["pruned"] = None
		return results
	
	
	
	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''
		
	def branchAndBound( self, time_allowance=60.0 ):
		# get set up
		cities = self._scenario.getCities()
		# get bssf
		greedy_result = self.greedy()
		bssf = greedy_result["cost"]
		num_solutions = 0
		num_pruned = 0
		num_states = 0
		num_updates = 0
		# initialize array
		start_array = [[math.inf for i in range(len(cities))] for j in range(len(cities))]
		for i in range(len(start_array)):
			for j in range(len(start_array[i])):
				start_array[i][j] = cities[i].costTo(cities[j])
		# start the clock
		start_time = time.time()

		# we are starting on the first city
		# set state of first city
		start_array, lower_bound = self._bound(start_array, True)
		best_route = [0]
		start_state = self.State(start_array, lower_bound, best_route)
		pq = hpq()
		pq.insert(start_state)

		# we are ready to start searching
		while time.time() - start_time < time_allowance and pq.size() > 0:

			# expand the next state
			# get the next on the queue
			next_state_expand = pq.delete_max()
			next_array = copy.deepcopy(next_state_expand.matrix)
			lower_bound = next_state_expand.lower_bound
			route_so_far = copy.deepcopy(next_state_expand.route_so_far)
			current_city = next_state_expand.get_current_city()
			# check if this state is a solution (len(route) == len(cities) + 1 && ?
			if len(route_so_far) == len(cities) + 1:
				num_solutions += 1
				# see if we have a route yet
				if best_route == [0]:
					best_route = copy.deepcopy(route_so_far)
				# check if this is new BSSF
				if lower_bound < bssf:
					bssf = lower_bound
					num_updates += 1
					best_route = copy.deepcopy(route_so_far)
					# check all states to prune
					num_pruned += pq.prune(bssf)

			# for every non inf place to go: make new state, bound new state, put state on queue
			for i in range(len(cities)):
				if next_array[current_city][i] == math.inf:
					continue
				num_states += 1
				new_matrix = copy.deepcopy(next_array)
				travel_cost = new_matrix[current_city][i]
				# set row and column costs to inf
				for j in range(len(cities)):
					new_matrix[current_city][j] = math.inf
					new_matrix[j][i] = math.inf
				# set to to from indice to inf
				new_matrix[i][current_city] = math.inf
				# bound new_matrix
				new_matrix, new_lower_bound = self._bound(new_matrix, False)
				new_lower_bound = new_lower_bound + lower_bound + travel_cost
				new_route = copy.deepcopy(route_so_far)
				new_route.append(i)
				new_state = self.State(new_matrix, new_lower_bound, new_route)
				# put state on queue
				if new_state.lower_bound > bssf:
					num_pruned += 1
				else:
					pq.insert(new_state)

		# translate indices to cities
		if best_route == [0]:
			results = {}
			results["cost"] = 0
			results["time"] = time.time() - start_time
			results["count"] = 0
			results["soln"] = None
			results["max"] = pq.max
			results["total"] = num_states
			results["pruned"] = num_pruned
		else:
			return_route = []
			for i in best_route:
				return_route.append(cities[i])
			return_route.pop(-1)
			bssf = TSPSolution(return_route)
			results = {}
			results["cost"] = bssf.cost
			results["time"] = time.time() - start_time
			results["count"] = num_solutions
			results["soln"] = bssf
			results["max"] = pq.max
			results["total"] = num_states
			results["pruned"] = num_pruned
		print(num_updates)
		return results

	def _bound(self, array, first):
		# given a 2x2 array, will compute the lower bound of the array
		# and return the array and the extra cost of the bounding.
		if first:
			cost = 0
			for i in range(len(array)):
				min = math.inf
				for j in range(len(array[i])):
					if array[i][j] < min:
						min = array[i][j]
				cost += min
				for j in range(len(array[i])):
					array[i][j] -= min
			for j in range(len(array)):
				min = math.inf
				for i in range(len(array[j])):
					if array[i][j] < min:
						min = array[i][j]
				cost += min
				for i in range(len(array[j])):
					array[i][j] -= min
			return array, cost
		else:
			cost = 0
			for j in range(len(array)):
				min = math.inf
				min_index = math.inf
				for i in range(len(array[j])):
					if array[i][j] < min:
						min = array[i][j]
						min_index = i
				if min_index != math.inf:
					cost += min
					array[min_index][j] -= min
			return array, cost

	class State:
		def __init__(self, matrix, lower_bound, route_so_far):
			self.matrix = matrix
			self.lower_bound = lower_bound
			self.route_so_far = route_so_far

		def get_current_city(self):
			return self.route_so_far[-1]

		def get_depth(self):
			return len(self.route_so_far)
	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''

	def fancy(self, time_allowance=60.0):
		# get set up
		cities = self._scenario.getCities()
		num_cities = len(cities)
		num_solutions = 0
		# initialize start array
		start_array = [[x.costTo(y) for y in cities] for x in cities]
		start_array = np.array(start_array)
		# initialize start degree array
		start_degree_array = [sum(1 for p in col if p != math.inf) for col in start_array.T]

		# define the gains of heuristic equation
		gain_cost = 1
		gain_degree = num_cities

		# get best solution so far if exists
		greedy_result = self.greedy()
		best_solution_length = greedy_result["cost"]
		best_solution_path = []
		# for testing the gains
		# best_solution_length = math.inf

		# start the clock
		start_time = time.time()
		end_time = math.inf

		start_city = 0
		current_path = [start_city]
		current_path_length = 0
		current_degree_array = start_degree_array.copy()
		# while we still have time
		while time.time() - start_time < time_allowance:
			current_city = start_city
			# until we have a circuit
			while len(current_path) < num_cities + 1:
				# if we have a not less worse path so far
				if current_path_length >= best_solution_length:
					break
				# if we return to start city
				if len(current_path) == num_cities:
					current_path.append(start_city)
					current_path_length += start_array[current_city][start_city]
					break

				# determine accessability array
				accessible = [i for i in range(len(start_array[current_city])) if start_array[current_city][i] != math.inf and i not in current_path]

				# if accessible is empty there is no solution
				if not accessible:
					current_path_length = math.inf
					break

				# get weighted costs
				C = self.get_costs(start_array[current_city], accessible)

				# weight the degrees
				D = current_degree_array.copy()
				length_temp = sum(current_degree_array[i] for i in accessible)

				# if 1 in D then must take it
				break_flag = False
				continue_flag = False
				for i in accessible:
					if D[i] == 1:
						current_path_length += start_array[current_city][i]
						# update current degree array
						current_degree_array[current_city] = math.inf
						for j in accessible:
							current_degree_array[j] -= 1
						# if a degree is zero we've lost
						if 0 in current_degree_array and len(current_path) != num_cities - 1:
							current_path_length = math.inf
							break_flag = True
							break
						# move to next city
						current_path.append(i)
						current_city = i
						continue_flag = True
						break
					D[i] = length_temp - D[i]
				if break_flag:
					break
				if continue_flag:
					continue
				# find next city
				current_city, current_path_length = self.find_next_city(C, D, accessible, current_city,
																		current_degree_array, current_path,
																		current_path_length, gain_cost, gain_degree,
																		num_cities, start_array)
			# if better solution is found
			if current_path_length < best_solution_length:
				best_solution_length = current_path_length
				best_solution_path = current_path
				num_solutions += 1
				end_time = time.time() - start_time
			# re-initialize stuff
			current_degree_array = start_degree_array.copy()
			start_city = (start_city + 1) % num_cities
			current_path = [start_city]
			current_path_length = 0

		# if greedy is better
		if not best_solution_path:
			results = {}
			results["cost"] = 0
			results["time"] = time.time() - start_time
			results["count"] = 0
			results["soln"] = None
			results["max"] = None
			results["total"] = None
			results["pruned"] = None
			return results
		# return the results
		return_route = []
		for i in best_solution_path:
			return_route.append(cities[i])
		return_route.pop(-1)
		bssf = TSPSolution(return_route)
		results = {}
		results["cost"] = bssf.cost
		results["time"] = end_time
		results["count"] = num_solutions
		results["soln"] = bssf
		results["max"] = None
		results["total"] = None
		results["pruned"] = None
		return results

	def find_next_city(self, C, D, accessible, current_city, current_degree_array, current_path, current_path_length,
					   gain_cost, gain_degree, num_cities, start_array):
		model = [gain_cost * C[i] + gain_degree * D[i] for i in accessible]
		length_temp = sum(p for p in model if p != math.inf)
		# generate random number
		ran_num = length_temp * rand.random()
		accumulation = 0
		# find next path to take
		for i in range(len(model)):
			accumulation += model[i]
			if accumulation >= ran_num:
				next_city = accessible[i]
				current_path_length += start_array[current_city][next_city]
				# update current degree array
				current_degree_array[current_city] = math.inf
				for j in accessible:
					current_degree_array[j] -= 1
				# if a degree is zero we've lost
				if 0 in current_degree_array and len(current_path) != num_cities - 1:
					current_path_length = math.inf
					break
				# move to next city
				current_path.append(next_city)
				current_city = next_city
				break
		return current_city, current_path_length

	def get_costs(self, intersection, accessible):
		# set up cost weights
		P = intersection.copy()

		# add up the costs
		length_temp = sum(P[i] for i in accessible)

		# weight the options
		for i in accessible:
			P[i] = length_temp - P[i]

		return P

	if __name__ == "__main__":
		a = [[1, 2, math.inf], [7, 2, 1], [math.inf, 9, 8]]
		a, cost = _bound(a, a)
		print(cost)
		print(a)

