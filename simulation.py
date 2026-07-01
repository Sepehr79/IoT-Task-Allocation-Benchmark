# Enviroment variables
NUM_SLOTS   = 100
NUM_EDGES   = 8
NUM_USERS   = 15 

SLOT_LENGHT = 0.1

USER_CONTEXT_SIZE = 500 # KB


# Task entity
class Task:
    def __init__(self, d, b, s, userIndex):
        self.userIndex = userIndex
        self.D = d # Required computation cycles
        self.B = b # Data size
        self.S = s # The time slot that user generates current task
        self.R = -1 # The time slot that this task is received by one of servers
        self.E = -1 # The time slot that this task is fully executed by a server
        self.A = [0 for i in range(NUM_EDGES)] # Selection vector 
        self.selected_edge_index = -1

    def select(self, num_edge):
        """Selects only one of the edge servers"""
        for i in range(len(self.A)):
            self.A[i] = 0
        self.A[num_edge - 1] = 1
        self.selected_edge_index = num_edge - 1


# Edge entity
class Edge:
    def __init__(self, f, r):
        self.F = f # Computation capacity based on cycle per seconds
        self.R = r # Data transmision capacity
        self.prev_queue = 0
        self.service_rate = self.F * SLOT_LENGHT


# User entity
class User:
    def __init__(self, c):
        self.C = c # Context size of the user
        self.last_selected_edge = -1


# Generate random task
def generate_task(time_slot, userIndex):
    # Required CPU cycles (Mega cycles)
    d = int(max(100, random.gauss(600, 150)))
    # Required Input data size (KB)
    b = int(max(10, random.gauss(500, 200)))
    return Task(d, b, time_slot, userIndex)

# Generate homogeneous edges
EDGE_CPUS = [ 4000 ] # MCycle/s
EDGE_RATES = [ 100 ] # Mb/s

def generate_edge():
    cpu = random.choice(EDGE_CPUS)
    data_rate = random.choice(EDGE_RATES)
    return Edge(cpu, data_rate)

def get_received_tasks(tasks, current_slot):
    return [task for task in tasks if task.R == current_slot]


## Simulation Algorithm
import random
import math
import numpy

users = [User(USER_CONTEXT_SIZE) for i in range(NUM_USERS)]
edges = [generate_edge() for i in range(NUM_EDGES)]
tasks = []

for slot in range(NUM_SLOTS):
    
    # All users may generate new task
    for userIndex in range(len(users)):
        if(random.randint(0, 100) < 80):
            task = generate_task(slot, userIndex)
            task.S = slot
            tasks.append(task)

    # Scheduler make decisions for all new tasks
    for task in tasks:
        if task.S == slot:
            selected_edge = random.randint(0, NUM_EDGES-1)
            task.select(selected_edge)
            selected_transmision_rate = numpy.dot(task.A, [edge.R for edge in edges])
            task.R = task.S + math.ceil(task.B / (selected_transmision_rate * SLOT_LENGHT))  
            current_user = users[task.userIndex]
            if (selected_edge != current_user.last_selected_edge):
                task.R += math.ceil(current_user.C / (selected_transmision_rate * SLOT_LENGHT))
            current_user.last_selected_edge = selected_edge

    # Get and process received tasks
    received_tasks = get_received_tasks(tasks, slot)
    queue_before = [edge.prev_queue for edge in edges]
    current_slot_queue = {}
    
    # Compute the time slot which the received tasks will be fully executed
    for task in received_tasks:
        selected_computation_capacity = numpy.dot(task.A, [edge.F for edge in edges])
        selected_edge_queue = numpy.dot(task.A, queue_before)
        waiting_cycles = current_slot_queue.get(task.selected_edge_index, 0)
        task.E = (
                   task.R +
                   math.ceil(task.D / (selected_computation_capacity * SLOT_LENGHT)) + 
                   math.ceil((waiting_cycles + selected_edge_queue) / (selected_computation_capacity * SLOT_LENGHT)) 
            )
        if task.selected_edge_index in current_slot_queue:
            current_slot_queue[task.selected_edge_index] += task.D
        else:
            current_slot_queue[task.selected_edge_index] = task.D
        
    
    # Update the edges' queues
    for i in range(len(edges)):
        new_work = current_slot_queue.get(i, 0)

        edges[i].prev_queue = (
            max(edges[i].prev_queue - edges[i].service_rate, 0)
            + new_work
        )
        

## Calculate the total delay
delay = 0
for task in tasks:
    if task.E != -1: # Select only finished tasks
        delay += task.E - task.S
print(delay)