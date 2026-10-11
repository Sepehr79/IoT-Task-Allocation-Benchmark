import numpy as np
import random
import math
from scipy.stats import truncnorm
from config import *
import copy
from itertools import product
import time
import pickle
from pathlib import Path
import re
import textwrap

NUM_EDGES = len(EDGE_CAPACITIES)

def generate_datarate():
    sigma = np.sqrt(
        np.log(
            1 + (DATA_RATE_STD / DATA_RATE_MEAN) ** 2
        )
    )
    mu = (
        np.log(DATA_RATE_MEAN)
        - 0.5 * sigma**2
    )
    return np.random.lognormal(
        mean=mu,
        sigma=sigma
    ) 


class Task:

    def __init__(self, d, b, s, user_index):

        self.userIndex = user_index

        self.D = d             
        self.B = b              

        self.S = s  # Generation slot
        self.R = -1 # Reception slot
        self.E = -1 # Completion slot

        self.A = [0 for _ in range(NUM_EDGES)]

        self.selected_edge_index = -1


        self.data_transmission_delay = 0.0
        self.context_transmission_delay = 0.0
        self.connection_delay = 0.0

        self.transmission_delay = 0.0

        self.queue_delay = 0.0

        self.processing_delay = 0.0

        self.total_delay = 0.0

        self.delays = [0 for _ in range(NUM_EDGES)]
        self.queues = [0 for _ in range(NUM_EDGES)]
        self.last_allocation = [0 for _ in range(NUM_EDGES)]
        self.user_rates = [0 for _ in range(NUM_EDGES)]



class Edge:

    def __init__(self, f):

        # CPU capacity in cycles/second
        self.F = f

        self.prev_queue = 0

        self.service_rate = (
            self.F * SLOT_LENGTH
        )


class User:

    def __init__(self, context_size):

        self.C = context_size

        self.last_selected_edge = -1

        self.connections = [generate_datarate() for _ in range(NUM_EDGES)]

        self.data = 0
        self.capacity = 0



def generate_task(time_slot, user_index):
    d = truncnorm.rvs(
        (PROCESSING_CAPACITY_MIN - PROCESSING_CAPACITY_MEAN) / PROCESSING_CAPACITY_STD,
        (PROCESSING_CAPACITY_MAX - PROCESSING_CAPACITY_MEAN) / PROCESSING_CAPACITY_STD,
        loc=PROCESSING_CAPACITY_MEAN,
        scale=PROCESSING_CAPACITY_STD
    )

    b = truncnorm.rvs(
        (DATA_SIZE_MIN - DATA_SIZE_MEAN) / DATA_SIZE_STD,
        (DATA_SIZE_MAX - DATA_SIZE_MEAN) / DATA_SIZE_STD,
        loc=DATA_SIZE_MEAN,
        scale=DATA_SIZE_STD
    )
        

    return Task(
        d,
        b,
        time_slot,
        user_index
    )


# Random scheduler
def random_schedule(tasks, edges, users, slot):

    for task in tasks:

        selected_edge = random.randrange(
            NUM_EDGES
        )

        task.A = [0] * NUM_EDGES

        task.A[selected_edge] = 1

        task.selected_edge_index = selected_edge

# Round-robin scheduler
def round_robin_schedule(tasks, edges, users, slot):

    edge_indices = list(range(NUM_EDGES))

    random.shuffle(edge_indices)

    for idx, task in enumerate(tasks):

        selected_edge = (
            edge_indices[idx % NUM_EDGES]
        )

        task.A = [0] * NUM_EDGES

        task.A[selected_edge] = 1

        task.selected_edge_index = selected_edge

def exhaustive_schedule(
    tasks,
    edges,
    users,
    slot
):

    start_time = time.perf_counter()

    num_tasks = len(tasks)
    num_edges = len(edges)

    if num_tasks == 0:
        return 0.0

    best_delay = float("inf")
    best_solution = None

    evaluated_solutions = 0


    for solution in product(
        range(num_edges),
        repeat=num_tasks
    ):

        cost = delay_cost(
            tasks,
            edges,
            users,
            slot,
            solution
        )

        evaluated_solutions += 1

        if cost < best_delay:

            best_delay = cost
            best_solution = list(solution)

    execution_time = (
        time.perf_counter() - start_time
    )

    for task_index, task in enumerate(tasks):

        edge_index = best_solution[task_index]

        task.A = [0] * num_edges

        task.A[edge_index] = 1

        task.selected_edge_index = edge_index

    # print(
    #     f"Exhaustive Search: "
    #     f"{evaluated_solutions} solutions evaluated, "
    #     f"time = {execution_time:.6f}s, "
    #     f"best delay = {best_delay:.6f}s"
    # )

    return best_delay

# Greedy scheduler
def greedy_context_schedule(tasks, edges, users, slot):

    temp_queue = [
        edge.prev_queue
        for edge in edges
    ]

    for task in tasks:

        best_edge = None
        best_cost = float("inf")

        user = users[task.userIndex]

        for edge_index, edge in enumerate(edges):

            transmission_rate = (
                task.user_rates[edge_index]
            )


            data_transmission = (
                task.B /
                transmission_rate
            )

            computation = (
                temp_queue[edge_index] +
                task.D
            ) / edge.F

            context = 0.0

            if (
                user.last_selected_edge
                != edge_index
            ):

                context_transmission = (
                    user.C /
                    transmission_rate
                )

                context = (
                    context_transmission +
                    WEBSOCKET_CONNECTION
                )

            cost = (
                data_transmission +
                computation +
                context
            )

            if cost < best_cost:

                best_cost = cost
                best_edge = edge_index

        task.A = [0] * NUM_EDGES

        task.A[best_edge] = 1

        task.selected_edge_index = best_edge

        temp_queue[best_edge] += task.D

import random
import math


# Cost of a complete assignment
def delay_cost(tasks, edges, users, slot, solution=None):

    temp_queue = [
        edge.prev_queue
        for edge in edges
    ]

    sum_delay = 0.0

    for task_index, task in enumerate(tasks):

        user = users[task.userIndex]

        if solution is not None:
            edge_index = solution[task_index]
        else:
            edge_index = task.selected_edge_index

        edge = edges[edge_index]

        transmission_rate = (
            task.user_rates[edge_index]
        )

        data_transmission = (
            task.B /
            transmission_rate
        )

        computation = (
            temp_queue[edge_index] +
            task.D
        ) / edge.F

        context = 0.0

        if user.last_selected_edge != edge_index:

            context_transmission = (
                user.C /
                transmission_rate
            )

            context = (
                context_transmission +
                WEBSOCKET_CONNECTION
            )

        cost = (
            data_transmission +
            computation +
            context
        )

        sum_delay += cost

        temp_queue[edge_index] += task.D

    return sum_delay


# Convert a WOA position into a valid discrete solution
def discretize_solution(position):

    solution = []

    for value in position:

        edge_index = int(round(value))

        edge_index = max(
            0,
            min(NUM_EDGES - 1, edge_index)
        )

        solution.append(edge_index)

    return solution


# Generate random solution
def random_solution(num_tasks):

    return [
        random.randint(0, NUM_EDGES - 1)
        for _ in range(num_tasks)
    ]


# Convert discrete solution to continuous WOA position
def solution_to_position(solution):

    return [
        float(edge_index)
        for edge_index in solution
    ]


# WOA scheduler
def woa_schedule(
    tasks,
    edges,
    users,
    slot,
    num_whales=20,
    max_iterations=100
):

    num_tasks = len(tasks)

    if num_tasks == 0:
        return 0.0

    # 1. Generate initial population
    population = []

    # First solution = Greedy solution
    greedy_context_schedule(
        tasks,
        edges,
        users,
        slot
    )

    greedy_solution = [
        task.selected_edge_index
        for task in tasks
    ]

    population.append(
        solution_to_position(greedy_solution)
    )

    # Remaining whales = random solutions
    for _ in range(num_whales - 1):

        solution = random_solution(num_tasks)

        population.append(
            solution_to_position(solution)
        )

    # 2. Evaluate initial population
    fitness = []

    for position in population:

        solution = discretize_solution(position)

        cost = delay_cost(
            tasks,
            edges,
            users,
            slot,
            solution
        )

        fitness.append(cost)

    # 3. Find initial best solution
    best_index = min(
        range(len(fitness)),
        key=lambda i: fitness[i]
    )

    best_position = population[best_index].copy()
    best_fitness = fitness[best_index]

    # 4. WOA main loop
    for iteration in range(max_iterations):

        # a decreases linearly from 2 to 0
        a = 2 - (
            2 * iteration / max_iterations
        )

        for i in range(num_whales):

            current_position = population[i]

            r1 = random.random()
            r2 = random.random()

            A = (
                2 * a * r1
            ) - a

            C = (
                2 * r2
            )

            p = random.random()

            l = random.uniform(-1, 1)

            b = 1.0

            new_position = [
                0.0
                for _ in range(num_tasks)
            ]

            # Case 1: Exploitation / encircling prey
            if p < 0.5:

                if abs(A) < 1:

                    # Encircling the current best solution
                    for j in range(num_tasks):

                        D = abs(
                            C * best_position[j]
                            - current_position[j]
                        )

                        new_position[j] = (
                            best_position[j]
                            - A * D
                        )

                else:

                    # Exploration:
                    # use a random whale
                    random_index = random.randint(
                        0,
                        num_whales - 1
                    )

                    random_position = population[
                        random_index
                    ]

                    for j in range(num_tasks):

                        D = abs(
                            C * random_position[j]
                            - current_position[j]
                        )

                        new_position[j] = (
                            random_position[j]
                            - A * D
                        )

            # Case 2: Bubble-net / spiral attack
            else:

                for j in range(num_tasks):

                    D = abs(
                        best_position[j]
                        - current_position[j]
                    )

                    new_position[j] = (
                        D *
                        math.exp(b * l) *
                        math.cos(2 * math.pi * l)
                        + best_position[j]
                    )

            # Keep position inside edge range
            for j in range(num_tasks):

                new_position[j] = max(
                    0.0,
                    min(
                        NUM_EDGES - 1,
                        new_position[j]
                    )
                )

            # Evaluate new whale
            new_solution = discretize_solution(
                new_position
            )

            new_fitness = delay_cost(
                tasks,
                edges,
                users,
                slot,
                new_solution
            )

            # Greedy-style elitist replacement
            current_solution = discretize_solution(
                current_position
            )

            current_fitness = delay_cost(
                tasks,
                edges,
                users,
                slot,
                current_solution
            )

            if new_fitness < current_fitness:

                population[i] = new_position
                fitness[i] = new_fitness

            # Update global best
            if new_fitness < best_fitness:

                best_position = new_position.copy()
                best_fitness = new_fitness


    best_solution = discretize_solution(
        best_position
    )

    # 6. Write final assignment into tasks
    for task_index, task in enumerate(tasks):

        edge_index = best_solution[task_index]

        task.A = [0] * NUM_EDGES

        task.A[edge_index] = 1

        task.selected_edge_index = edge_index

    # 7. Return objective value
    return best_fitness

def greedy_schedule(tasks, edges, users, slot):


    temp_queue = [
        edge.prev_queue
        for edge in edges
    ]

    for task in tasks:

        best_edge = None
        best_cost = float("inf")

        user = users[task.userIndex]

        for edge_index, edge in enumerate(edges):

            transmission_rate = (
                task.user_rates[edge_index]
            )

            data_transmission = (
                task.B /
                transmission_rate
            )

            computation = (
                temp_queue[edge_index] +
                task.D
            ) / edge.F

            cost = (
                data_transmission +
                computation
            )

            if cost < best_cost:

                best_cost = cost
                best_edge = edge_index

        task.A = [0] * NUM_EDGES

        task.A[best_edge] = 1

        task.selected_edge_index = best_edge

        temp_queue[best_edge] += task.D


scheduling = {
    "r": random_schedule,
    "rr": round_robin_schedule,
    "g": greedy_schedule,
    "gc": greedy_context_schedule,
    "woa": woa_schedule,
    "exh": exhaustive_schedule,
}

users = [
    User(USER_CONTEXT_SIZE)
    for _ in range(NUM_USERS)
]

edges = [ Edge(f) for f in EDGE_CAPACITIES ]


def calc_receive_time(task):
        user = users[task.userIndex]
        transmission_rate = (
                    task.user_rates[task.selected_edge_index]
                )
        task.data_transmission_delay = (
                    task.B /
                    transmission_rate
        )
        if (task.selected_edge_index != user.last_selected_edge):
            task.context_transmission_delay = (
                user.C /
                transmission_rate
            )
            task.connection_delay = (WEBSOCKET_CONNECTION)
            user.last_selected_edge = (task.selected_edge_index)
        else:
            task.context_transmission_delay = 0.0
            task.connection_delay = 0.0

        task.transmission_delay = (
                    task.data_transmission_delay +
                    task.context_transmission_delay
                )
        communication_delay = (
                    task.transmission_delay +
                    task.connection_delay
                )
        communication_slots = math.ceil(
                    communication_delay /
                    SLOT_LENGTH
                )
        task.R = (task.S + communication_slots)

# Save results to a file named after the algorithm and workload
def _safe(name) -> str:
    return re.sub(r"[^\w.-]+", "_", str(name)).strip("_")

schedule = scheduling[SCHEDULE_ALGORITHM]

tasks = []

server_changed = 0

slot_tasks = {}

if __name__ == "__main__":

    prev_tasks = []
    with open("slot_tasks.pkl", "rb") as file:
        prev_tasks = pickle.load(file)

    # Main simulation
    for slot in range(
        NUM_SLOTS + EXTRA_SLOTS
    ):

        queue_before = [
            edge.prev_queue
            for edge in edges
        ]

        if slot < NUM_SLOTS:

            if not TASK_GENERATION and len(prev_tasks) > 0:
                for task in prev_tasks[slot]:

                    task = copy.deepcopy(task)

                    # Current execution state
                    task.queues = queue_before.copy()

                    user_index = task.userIndex

                    last_edge = users[user_index].last_selected_edge

                    if last_edge != -1:
                        task.last_allocation[last_edge] = 1

                    # Add task to current simulation
                    tasks.append(task)

                    users[user_index].data += task.B
                    users[user_index].capacity += task.D
                
            else:
                slot_tasks[slot] = []
                for user_index in range(NUM_USERS):
                    if (
                        random.random()
                        <
                        USER_TASK_GENERATION_PROBABILITY
                    ):
                        
                        task = generate_task(
                            slot,
                            user_index
                        )
                        task.queues = queue_before.copy()
                        last_edge = users[user_index].last_selected_edge
                        task.user_rates = users[user_index].connections.copy()
                        if last_edge != -1:
                            task.last_allocation[last_edge] = 1
                        tasks.append(task)
                        slot_tasks[slot].append(copy.deepcopy(task))
                        users[user_index].data += task.B
                        users[user_index].capacity += task.D
            


        arrived_tasks = [
            task for task in tasks
            if task.S == slot
        ]

        print(f"Processing slot: {slot}, Number of tasks: {len(arrived_tasks)}", end="")


        start_time = 0
        end_time = 0
        if arrived_tasks:

            start_time = time.perf_counter()
            # Then perform normal allocation
            schedule(
                arrived_tasks,
                edges,
                users,
                slot
            )
            end_time = time.perf_counter()

        print(f", Elapsed time: {(end_time - start_time) * 1000:.3f} ms")

        for task in arrived_tasks:
            user = users[task.userIndex]
            if task.selected_edge_index != user.last_selected_edge:
                server_changed += 1
            calc_receive_time(
                task,
            )

            
        current_slot_queue = {}


        for task in tasks:

            if (
                task.R == slot
                and
                task.E == -1
            ):

                edge_index = (
                    task.selected_edge_index
                )

                edge = edges[edge_index]

                previous_queue_cycles = (
                    queue_before[edge_index]
                )


                current_queue_cycles = (
                    current_slot_queue.get(
                        edge_index,
                        0
                    )
                )

                waiting_cycles = (
                    previous_queue_cycles +
                    current_queue_cycles
                )


                task.queue_delay = (
                    waiting_cycles /
                    edge.F
                )

                task.processing_delay = (
                    task.D /
                    edge.F
                )

                queue_slots = math.ceil(
                    task.queue_delay /
                    SLOT_LENGTH
                )

                processing_slots = math.ceil(
                    task.processing_delay /
                    SLOT_LENGTH
                )

                task.E = (
                    task.R +
                    queue_slots +
                    processing_slots
                )

                task.total_delay = (
                    task.transmission_delay +
                    task.connection_delay +
                    task.queue_delay +
                    task.processing_delay
                )

                current_slot_queue[
                    edge_index
                ] = (
                    current_queue_cycles +
                    task.D
                )


        for edge_index in range(NUM_EDGES):

            new_work = (
                current_slot_queue.get(
                    edge_index,
                    0
                )
            )

            edges[edge_index].prev_queue = (
                max(
                    edges[edge_index].prev_queue
                    -
                    edges[edge_index].service_rate,
                    0
                )
                +
                new_work
            )

        for user in users:
            user.connections = [generate_datarate() for _ in range(NUM_EDGES)]

    completed_tasks = [
        task
        for task in tasks
        if task.E != -1
    ]

    number_of_completed_tasks = len(completed_tasks)

    # Workload is independent of completed tasks
    workload = (
        USER_TASK_GENERATION_PROBABILITY
        * NUM_USERS
        * PROCESSING_CAPACITY_MEAN
    ) / (
        sum(edge.F for edge in edges)
        * SLOT_LENGTH
    )

    # Default values
    average_data_transmission = 0.0
    average_context_transmission = 0.0
    average_transmission = 0.0
    average_connection = 0.0
    average_queue = 0.0
    average_processing = 0.0
    average_total = 0.0
    if number_of_completed_tasks > 0:

        total_data_transmission = sum(
            task.data_transmission_delay
            for task in completed_tasks
        )

        total_context_transmission = sum(
            task.context_transmission_delay
            for task in completed_tasks
        )

        total_transmission = sum(
            task.transmission_delay
            for task in completed_tasks
        )

        total_connection = sum(
            task.connection_delay
            for task in completed_tasks
        )

        total_queue = sum(
            task.queue_delay
            for task in completed_tasks
        )

        total_processing = sum(
            task.processing_delay
            for task in completed_tasks
        )

        total_delay = sum(
            task.total_delay
            for task in completed_tasks
        )

        average_data_transmission = (
            total_data_transmission /
            number_of_completed_tasks
        )

        average_context_transmission = (
            total_context_transmission /
            number_of_completed_tasks
        )

        average_transmission = (
            total_transmission /
            number_of_completed_tasks
        )

        average_connection = (
            total_connection /
            number_of_completed_tasks
        )

        average_queue = (
            total_queue /
            number_of_completed_tasks
        )

        average_processing = (
            total_processing /
            number_of_completed_tasks
        )

        average_total = (
            total_delay /
            number_of_completed_tasks
        )



    # ============================================================
    # Save and print results
    # ============================================================

    results = f"""
    {"=" * 55}
                    DELAY STATISTICS
    {"=" * 55}
    Algorithm: {schedule.__name__}
    Workload: {workload}
    Completed tasks: {number_of_completed_tasks}
    {"-" * 55}
    Average data transmission delay: {average_data_transmission:.4f} s
    Average context transmission delay: {average_context_transmission:.4f} s
    Average transmission delay: {average_transmission:.4f} s
    Average connection/setup delay: {average_connection:.4f} s
    Average queue delay: {average_queue:.4f} s
    Average processing delay: {average_processing:.4f} s
    {"-" * 55}
    Average total delay: {average_total:.4f} s
    Server changes: {server_changed}
    {"=" * 55}
    """
    # Print to console
    print(results)

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    result_path = output_dir / f"{_safe(schedule.__name__)}_{_safe(workload)}.txt"
    result_path.write_text(textwrap.dedent(results).strip() + "\n", encoding="utf-8")
    print(f"Results saved to {result_path}")

    if len(slot_tasks) > 0:
        with open("slot_tasks.pkl", "wb") as file:
                pickle.dump(slot_tasks, file)
                print("slot_tasks saved successfully.")