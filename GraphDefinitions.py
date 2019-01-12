def state_name_to_int(state):
    state_name_map = {
        'S': 0,
        'A': 1,
        'B': 2,
        'C': 3,
        'D': 4,
        'E': 5,
        'F': 6,
        'G': 7,
        'H': 8,
        'K': 9,
        'L': 10,
        'M': 11,
        'N': 12,
        'O': 13
    }
    return state_name_map[state]

def int_to_state_name(state_as_int):
    state_map = {
        0: 'S',
        1: 'A',
        2: 'B',
        3: 'C',
        4: 'D',
        5: 'E',
        6: 'F',
        7: 'G',
        8: 'H',
        9: 'K',
        10: 'L',
        11: 'M',
        12: 'N',
        13: 'O'
    }
    return state_map[state_as_int]


def getSmallGraph():
    return {
            'S': [('A', 100), ('B', 400), ('C', 200 )],
            'A': [('B', 250), ('C', 400), ('S', 100 )],
            'B': [('A', 250), ('C', 250), ('S', 400 )],
            'C': [('A', 400), ('B', 250), ('S', 200 )]
        }

def getLargeGraph():
    return {
            'S': [('A', 300), ('B', 100), ('C', 200 )],
            'A': [('S', 300), ('B', 100), ('E', 100 ), ('D', 100 )],
            'B': [('S', 100), ('A', 100), ('C', 50 ), ('K', 200 )],
            'C': [('S', 200), ('B', 50), ('M', 100 ), ('L', 200 )],
            'D': [('A', 100), ('F', 50)],
            'E': [('A', 100), ('F', 100), ('H', 100)],
            'F': [('D', 50), ('E', 100), ('G', 200)],
            'G': [('F', 200), ('O', 300)],
            'H': [('E', 100), ('K', 300)],
            'K': [('B', 200), ('H', 300)],
            'L': [('C', 200), ('M', 50)],
            'M': [('C', 100), ('L', 50), ('N', 100)],
            'N': [('M', 100), ('O', 100)],
            'O': [('N', 100), ('G', 300)]
        }

def createDistanceMatrix( graph ):
    size = len(graph)
    result = {}

    for from_node in range(size):
        result[from_node] = {}
        from_name = int_to_state_name(from_node)
        for to_node in range(size):
            to_name = int_to_state_name(to_node)
            distance = getDistance(from_name,to_name,graph)
            if distance == 0:
                distance = getDistance(to_name,from_name,graph)
            result[from_node][to_node] = distance
    return result

def getDistance( from_name, to_name, map ):
    result = 0
    if from_name in map:
        fromlist = map[from_name]
        for distance in fromlist:
            if distance[0] == to_name:
                result = distance[1]
                break
    return result

print( createDistanceMatrix(getSmallGraph()))


from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

###########################
# Problem Data Definition #
###########################
def create_data_model(graph,customer_rewards):
  """Creates the data for the example."""
  data = {}
  # Array of distances between locations.
  _distances = createDistanceMatrix(graph)
  demands = customer_rewards
  capacities = [1000000] 
  data["distances"] = _distances
  data["num_locations"] = len(_distances)
  data["num_vehicles"] = 1
  data["depot"] = 0
  data["demands"] = demands
  data["vehicle_capacities"] = capacities
  return data

#######################
# Problem Constraints #
#######################
def create_distance_callback(data):
  """Creates callback to return distance between points."""
  distances = data["distances"]

  def distance_callback(from_node, to_node):
    """Returns the manhattan distance between the two nodes"""
    return distances[from_node][to_node]
  return distance_callback

def create_demand_callback(data):
    """Creates callback to get demands at each location."""
    def demand_callback(from_node, to_node):
        return data["demands"][from_node]
    return demand_callback

def add_capacity_constraints(routing, data, demand_callback):
    """Adds capacity constraint"""
    capacity = "Capacity"
    routing.AddDimensionWithVehicleCapacity(
        demand_callback,
        0, # null capacity slack
        data["vehicle_capacities"], # vehicle maximum capacities
        True, # start cumul to zero
        capacity)

def getTotalReward(graph,customer_reward):
    result = 0
    # Instantiate the data problem.
    data = create_data_model(graph,customer_reward)
    # Create Routing Model
    routing = pywrapcp.RoutingModel(
        data["num_locations"],
        data["num_vehicles"],
        data["depot"])
    # Define weight of each edge
    distance_callback = create_distance_callback(data)
    routing.SetArcCostEvaluatorOfAllVehicles(distance_callback)
    # Add Capacity constraint
    demand_callback = create_demand_callback(data)
    add_capacity_constraints(routing, data, demand_callback)
    # Setting first solution heuristic (cheapest addition).
    search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
    # search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters)
    if assignment:
        result = getAssignmentReward(data, routing, assignment)  
    else:
        print("OR tools: no solution found")  
    return result

###########
# Printer #
###########
def getAssignmentReward(data, routing, assignment):
    result = 0
    """Print routes on console."""
    total_dist = 0
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {0}:\n'.format(vehicle_id)
        route_dist = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = routing.IndexToNode(index)
            next_node_index = routing.IndexToNode(assignment.Value(routing.NextVar(index)))
            route_dist += routing.GetArcCostForVehicle(node_index, next_node_index, vehicle_id)
            route_load += data["demands"][node_index]
            plan_output += ' {0} Load({1}) -> '.format(node_index, route_load)
            index = assignment.Value(routing.NextVar(index))

        result -= route_dist
        result += route_load
        node_index = routing.IndexToNode(index)
        total_dist += route_dist
        plan_output += ' {0} Load({1})\n'.format(node_index, route_load)
        plan_output += 'Distance of the route: {0}m\n'.format(route_dist)
        plan_output += 'Load of the route: {0}\n'.format(route_load)
        print(plan_output)
    print('Total reward: {0}'.format(result))
    return result

graph = getSmallGraph()
rewards = [1000]*len(graph)
print(getTotalReward(graph,rewards))
