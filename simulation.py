# Enviroment variables
NUM_SLOTS   = 200
NUM_EDGES   = 8
NUM_USERS   = 15 

SLOT_LENGHT = 2 # Seconds
USER_TASK_GENERATION_PROBABILITY = 0.5

USER_CONTEXT_SIZE = 500 # KB

EDGE_CPUS = [ 4*10**9 ] # 4GHz
TRANSMISON_RATES = [5000 for _ in range(NUM_EDGES)]  # 5000 KB/s = 5 MB/s starting transmission rates between all users and edges


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


# Edge entity
class Edge:
    def __init__(self, f):
        self.F = f # Computation capacity based on cycle per seconds
        self.prev_queue = 0
        self.service_rate = self.F * SLOT_LENGHT


# User entity
class User:
    def __init__(self, c):
        self.C = c # Context size of the user
        self.last_selected_edge = -1
        self.connections = TRANSMISON_RATES.copy()


# Generate random task
def generate_task(time_slot, userIndex):
    d = int(random.randint(600*10**6, 1200*10**6))
    # Required Input data size (KB)
    b = int(random.randint(500, 3000))
    return Task(d, b, time_slot, userIndex)

# Generate homogeneous edges
def generate_edge():
    cpu = random.choice(EDGE_CPUS)
    return Edge(cpu)


def schedule(task, edges, users, slot):
    """Selects only one of the edge servers"""
    selected_edge = random.randint(0, NUM_EDGES-1)
    for i in range(len(task.A)):
        task.A[i] = 0
    task.A[selected_edge] = 1
    task.selected_edge_index = selected_edge

## Simulation Algorithm
import random
import math
import numpy

users = [User(USER_CONTEXT_SIZE) for i in range(NUM_USERS)]
edges = [generate_edge() for i in range(NUM_EDGES)]
tasks = []

computation_capacity_per_slot = sum([edge.F for edge in edges]) * SLOT_LENGHT 

for slot in range(NUM_SLOTS):

    # Task generation   
    for user_index in range(NUM_USERS):
        if random.random() < USER_TASK_GENERATION_PROBABILITY:
            task = generate_task(slot, user_index)
            task.S = slot
            tasks.append(task)


    # Scheduler make decisions for all new tasks
    for task in tasks:
        if task.S == slot:
            schedule(task, edges, users, slot)
            selected_transmision_rate = numpy.dot(task.A, users[task.selected_edge_index].connections)
            task.R = task.S + math.ceil(task.B / (selected_transmision_rate * SLOT_LENGHT))  
            current_user = users[task.userIndex]
            if (task.selected_edge_index != current_user.last_selected_edge):
                task.R += math.ceil(current_user.C / (selected_transmision_rate * SLOT_LENGHT))
            current_user.last_selected_edge = task.selected_edge_index

    # Get and process received tasks
    queue_before = [edge.prev_queue for edge in edges]
    current_slot_queue = {}
    
    # Compute the time slot which the received tasks will be fully executed
    for task in tasks:
        if task.R == slot:
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