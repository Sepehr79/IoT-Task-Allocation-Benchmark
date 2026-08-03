# Enviroment variables
NUM_SLOTS   = 200
NUM_EDGES   = 8
NUM_USERS   = 15 

SLOT_LENGHT = 0.47 # Seconds
USER_TASK_GENERATION_PROBABILITY = 0.8

USER_CONTEXT_SIZE = 200 # KB

EDGE_CPUS = [ 2*10**9 ] # 2GHz
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
    d = int(random.randint(10**8, 10**9))
    # Required Input data size (KB)
    b = int(random.randint(100, 500))
    return Task(d, b, time_slot, userIndex)

# Generate homogeneous edges
def generate_edge():
    cpu = random.choice(EDGE_CPUS)
    return Edge(cpu)


def schedule(tasks, edges, users, slot):
    """Randomly distribute tasks among edge servers"""
    edge_indices = list(range(NUM_EDGES))
    random.shuffle(edge_indices)
    for idx, task in enumerate(tasks):
        selected_edge = edge_indices[idx % NUM_EDGES]

        task.A = [0] * NUM_EDGES
        task.A[selected_edge] = 1
        task.selected_edge_index = selected_edge

def greedy_schedule(tasks, edges, users, slot):
    temp_queue = [edge.prev_queue for edge in edges]
    for task in tasks:
        best = None
        best_cost = float('inf')
        for i, edge in enumerate(edges):
            transmission = (
                task.B /
                users[task.userIndex].connections[i]
            )
            computation = (
                temp_queue[i] + task.D
            ) / edge.F
            context = 0
            if users[task.userIndex].last_selected_edge != i:
                context = (
                    users[task.userIndex].C /
                    users[task.userIndex].connections[i]
                )
            cost = transmission + computation + context
            if cost < best_cost:
                best_cost = cost
                best = i

        task.A = [0]*NUM_EDGES
        task.A[best] = 1
        task.selected_edge_index = best
        temp_queue[best] += task.D


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

    # Get and process received tasks
    queue_before = [edge.prev_queue for edge in edges]
    current_slot_queue = {}
    arrived_tasks = []

    for task in tasks:
        # Gathering all new tasks to make decision for all at once
        if task.S == slot:
            arrived_tasks.append(task)

        # Work out the time slot which the received tasks will be fully executed
        if task.R == slot:
            selected_computation_capacity = edges[task.selected_edge_index].F
            selected_edge_queue = queue_before[task.selected_edge_index]
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

    # Schedule all new tasks at once
    schedule(arrived_tasks, edges, users, slot)
    # Work out the time slot in which task is received by the server
    for task in arrived_tasks:
        selected_transmision_rate = users[task.userIndex].connections[task.selected_edge_index]
        task.R = task.S + math.ceil(task.B / (selected_transmision_rate * SLOT_LENGHT))  
        current_user = users[task.userIndex]
        if (task.selected_edge_index != current_user.last_selected_edge):
            task.R += math.ceil(current_user.C / (selected_transmision_rate * SLOT_LENGHT))
            current_user.last_selected_edge = task.selected_edge_index
        


    
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