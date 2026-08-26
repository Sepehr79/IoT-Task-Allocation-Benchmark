import numpy as np
import random
import math
from scipy.stats import truncnorm


# Enviroment variables
NUM_SLOTS = 80_000
EXTRA_SLOTS = 20_000
NUM_EDGES = 8
NUM_USERS = 15
WEBSOCKET_CONNECTION = 2.0  # seconds
PROCESSING_CAPACITY_MEAN = 1e9
PROCESSING_CAPACITY_MIN  = 0.5e9
PROCESSING_CAPACITY_MAX  = 1.5e9
PROCESSING_CAPACITY_STD  = 0.15e9

DATA_SIZE_MEAN = 120      # KB
DATA_SIZE_MIN  = 80       # KB
DATA_SIZE_MAX  = 450      # KB
DATA_SIZE_STD  = 40       # KB

DATA_RATE_MEAN = 2000       # KB/s
DATA_RATE_STD = 300         # KB/s

SLOT_LENGTH = 0.03  # seconds = 30 ms
USER_TASK_GENERATION_PROBABILITY = 0.0255

USER_CONTEXT_SIZE = 60  # KB
EDGE_CPUS = [
    1 * 10**9,
    2 * 10**9,
    2 * 10**9,
    2 * 10**9,
    3 * 10**9
]

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

def generate_edge():

    cpu = random.choice(EDGE_CPUS)

    return Edge(cpu)


# Round-robin scheduler
def rr_schedule(tasks, edges, users, slot):

    edge_indices = list(range(NUM_EDGES))

    random.shuffle(edge_indices)

    for idx, task in enumerate(tasks):

        selected_edge = (
            edge_indices[idx % NUM_EDGES]
        )

        task.A = [0] * NUM_EDGES

        task.A[selected_edge] = 1

        task.selected_edge_index = selected_edge


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
                user.connections[edge_index]
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
                user.connections[edge_index]
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



users = [
    User(USER_CONTEXT_SIZE)
    for _ in range(NUM_USERS)
]

edges = [
    generate_edge()
    for _ in range(NUM_EDGES)
]

tasks = []

server_changed = 0


# Main simulation
for slot in range(
    NUM_SLOTS + EXTRA_SLOTS
):
    print(f"Processing slot: {slot}")


    if slot < NUM_SLOTS:

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

                tasks.append(task)


    queue_before = [
        edge.prev_queue
        for edge in edges
    ]


    arrived_tasks = [
        task
        for task in tasks
        if task.S == slot
    ]



    if arrived_tasks:

        greedy_schedule(
            arrived_tasks,
            edges,
            users,
            slot
        )


    for task in arrived_tasks:

        user = users[task.userIndex]

        edge_index = (
            task.selected_edge_index
        )

        transmission_rate = (
            user.connections[edge_index]
        )


        task.data_transmission_delay = (
            task.B /
            transmission_rate
        )

        if (
            edge_index
            !=
            user.last_selected_edge
        ):

            task.context_transmission_delay = (
                user.C /
                transmission_rate
            )

            task.connection_delay = (
                WEBSOCKET_CONNECTION
            )

            server_changed += 1

            user.last_selected_edge = (
                edge_index
            )

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


        task.R = (
            task.S +
            communication_slots
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


number_of_completed_tasks = (
    len(completed_tasks)
)


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

    workload = (
        (USER_TASK_GENERATION_PROBABILITY * NUM_USERS * PROCESSING_CAPACITY_MEAN) /
        (sum(edge.F for edge in edges) * SLOT_LENGTH)
    )


    # Print results
    print("\n")
    print("=" * 55)
    print("                 DELAY STATISTICS")
    print("=" * 55)

    print(
        f"Workload: {workload}"
    )

    print(
        f"Completed tasks: "
        f"{number_of_completed_tasks}"
    )

    print("-" * 55)

    print(
        f"Average data transmission delay: "
        f"{average_data_transmission:.4f} s"
    )

    print(
        f"Average context transmission delay: "
        f"{average_context_transmission:.4f} s"
    )

    print(
        f"Average transmission delay: "
        f"{average_transmission:.4f} s"
    )

    print(
        f"Average connection/setup delay: "
        f"{average_connection:.4f} s"
    )

    print(
        f"Average queue delay: "
        f"{average_queue:.4f} s"
    )

    print(
        f"Average processing delay: "
        f"{average_processing:.4f} s"
    )

    print("-" * 55)

    print(
        f"Average total delay: "
        f"{average_total:.4f} s"
    )

    print(
        f"Server changes: "
        f"{server_changed}"
    )

    print("=" * 55)


else:

    print("No completed tasks.")

