# ============================================================
# Environment variables
# ============================================================

NUM_SLOTS = 80_000
EXTRA_SLOTS = 20_000

NUM_EDGES = 8
NUM_USERS = 15

# Connection/setup delay when switching to another edge
WEBSOCKET_CONNECTION = 2.0  # seconds

# Mean required computation
PROCESSING_CAPACITY_MEAN = 5 * 10**8  # cycles

# Simulation slot
SLOT_LENGTH = 0.03  # seconds = 30 ms

# Task generation probability per user per slot
USER_TASK_GENERATION_PROBABILITY = 0.0126

# User context size
USER_CONTEXT_SIZE = 60  # KB

# Heterogeneous edge CPU capacities
EDGE_CPUS = [
    2 * 10**8,
    4 * 10**8,
    6 * 10**8,
    8 * 10**8
]

# Transmission rate between every user and edge
TRANSMISSION_RATES = [
    2000 for _ in range(NUM_EDGES)
]  # KB/s


# ============================================================
# Imports
# ============================================================

import numpy as np
import random
import math


# ============================================================
# Task entity
# ============================================================

class Task:

    def __init__(self, d, b, s, user_index):

        # -------------------------
        # Task information
        # -------------------------

        self.userIndex = user_index

        self.D = d              # Required computation cycles
        self.B = b              # Input data size (KB)

        self.S = s              # Generation slot
        self.R = -1             # Reception slot
        self.E = -1             # Completion slot

        # -------------------------
        # Edge selection
        # -------------------------

        self.A = [0 for _ in range(NUM_EDGES)]

        self.selected_edge_index = -1

        # -------------------------
        # Delay components
        # -------------------------

        self.data_transmission_delay = 0.0
        self.context_transmission_delay = 0.0
        self.connection_delay = 0.0

        self.transmission_delay = 0.0

        self.queue_delay = 0.0

        self.processing_delay = 0.0

        self.total_delay = 0.0


# ============================================================
# Edge entity
# ============================================================

class Edge:

    def __init__(self, f):

        # CPU capacity in cycles/second
        self.F = f

        # Queued computation workload in cycles
        self.prev_queue = 0

        # Computation service capacity per slot
        self.service_rate = (
            self.F * SLOT_LENGTH
        )


# ============================================================
# User entity
# ============================================================

class User:

    def __init__(self, context_size):

        # Context size in KB
        self.C = context_size

        # Last selected edge
        self.last_selected_edge = -1

        # Transmission rates to all edges
        self.connections = TRANSMISSION_RATES.copy()


# ============================================================
# Generate random task
# ============================================================

def generate_task(time_slot, user_index):

    # Required computation cycles
    d = np.random.exponential(
        scale=PROCESSING_CAPACITY_MEAN
    )

    # Required input data size (KB)
    b = np.random.exponential(scale=120)
        

    return Task(
        d,
        b,
        time_slot,
        user_index
    )


# ============================================================
# Generate edge
# ============================================================

def generate_edge():

    cpu = random.choice(EDGE_CPUS)

    return Edge(cpu)


# ============================================================
# Round-robin scheduler
# ============================================================

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


# ============================================================
# Greedy scheduler
# ============================================================

def greedy_context_schedule(tasks, edges, users, slot):

    # Temporary queues used for estimating
    # the effect of scheduling current tasks
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

            # --------------------------------
            # Data transmission time
            # --------------------------------

            data_transmission = (
                task.B /
                transmission_rate
            )

            # --------------------------------
            # Estimated queue + processing
            # --------------------------------

            computation = (
                temp_queue[edge_index] +
                task.D
            ) / edge.F

            # --------------------------------
            # Context/setup cost
            # --------------------------------

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

            # --------------------------------
            # Total estimated cost
            # --------------------------------

            cost = (
                data_transmission +
                computation +
                context
            )

            if cost < best_cost:

                best_cost = cost
                best_edge = edge_index

        # --------------------------------
        # Assign selected edge
        # --------------------------------

        task.A = [0] * NUM_EDGES

        task.A[best_edge] = 1

        task.selected_edge_index = best_edge

        # --------------------------------
        # Update temporary queue
        # --------------------------------

        temp_queue[best_edge] += task.D


def greedy_schedule(tasks, edges, users, slot):

    # Temporary queues used for estimating
    # the effect of scheduling current tasks
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

            # --------------------------------
            # Data transmission time
            # --------------------------------

            data_transmission = (
                task.B /
                transmission_rate
            )

            # --------------------------------
            # Estimated queue + processing
            # --------------------------------

            computation = (
                temp_queue[edge_index] +
                task.D
            ) / edge.F

            # --------------------------------
            # Context/setup cost
            # --------------------------------

            # --------------------------------
            # Total estimated cost
            # --------------------------------

            cost = (
                data_transmission +
                computation
            )

            if cost < best_cost:

                best_cost = cost
                best_edge = edge_index

        # --------------------------------
        # Assign selected edge
        # --------------------------------

        task.A = [0] * NUM_EDGES

        task.A[best_edge] = 1

        task.selected_edge_index = best_edge

        # --------------------------------
        # Update temporary queue
        # --------------------------------

        temp_queue[best_edge] += task.D


# ============================================================
# Initialize simulation
# ============================================================

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


# ============================================================
# Main simulation
# ============================================================

for slot in range(
    NUM_SLOTS + EXTRA_SLOTS
):
    print(f"Processing slot: {slot}")


    # --------------------------------
    # Generate new tasks
    # --------------------------------

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


    # --------------------------------
    # Save queue state
    # --------------------------------

    queue_before = [
        edge.prev_queue
        for edge in edges
    ]


    # --------------------------------
    # Find tasks generated now
    # --------------------------------

    arrived_tasks = [
        task
        for task in tasks
        if task.S == slot
    ]


    # --------------------------------
    # Schedule new tasks
    # --------------------------------

    if arrived_tasks:

        greedy_context_schedule(
            arrived_tasks,
            edges,
            users,
            slot
        )


    # ========================================================
    # Transmission phase
    # ========================================================

    for task in arrived_tasks:

        user = users[task.userIndex]

        edge_index = (
            task.selected_edge_index
        )

        transmission_rate = (
            user.connections[edge_index]
        )


        # --------------------------------
        # 1. Input data transmission
        # --------------------------------

        task.data_transmission_delay = (
            task.B /
            transmission_rate
        )


        # --------------------------------
        # 2. Context transmission
        # --------------------------------

        if (
            edge_index
            !=
            user.last_selected_edge
        ):

            task.context_transmission_delay = (
                user.C /
                transmission_rate
            )

            # --------------------------------
            # 3. Connection/setup delay
            # --------------------------------

            task.connection_delay = (
                WEBSOCKET_CONNECTION
            )

            server_changed += 1

            # Update user's current edge
            user.last_selected_edge = (
                edge_index
            )

        else:

            task.context_transmission_delay = 0.0

            task.connection_delay = 0.0


        # --------------------------------
        # Pure transmission delay
        # --------------------------------

        task.transmission_delay = (
            task.data_transmission_delay +
            task.context_transmission_delay
        )


        # --------------------------------
        # Communication delay including
        # connection setup
        # --------------------------------

        communication_delay = (
            task.transmission_delay +
            task.connection_delay
        )


        # --------------------------------
        # Convert communication time
        # to simulation slots
        # --------------------------------

        communication_slots = math.ceil(
            communication_delay /
            SLOT_LENGTH
        )


        # --------------------------------
        # Reception slot
        # --------------------------------

        task.R = (
            task.S +
            communication_slots
        )


    # ========================================================
    # Processing phase
    # ========================================================

    current_slot_queue = {}


    for task in tasks:

        # --------------------------------
        # Task has reached its edge
        # --------------------------------

        if (
            task.R == slot
            and
            task.E == -1
        ):

            edge_index = (
                task.selected_edge_index
            )

            edge = edges[edge_index]


            # --------------------------------
            # Existing queue
            # --------------------------------

            previous_queue_cycles = (
                queue_before[edge_index]
            )


            # --------------------------------
            # Tasks already arriving at this
            # edge during this slot
            # --------------------------------

            current_queue_cycles = (
                current_slot_queue.get(
                    edge_index,
                    0
                )
            )


            # --------------------------------
            # Total workload ahead
            # --------------------------------

            waiting_cycles = (
                previous_queue_cycles +
                current_queue_cycles
            )


            # --------------------------------
            # Queue delay
            # --------------------------------

            task.queue_delay = (
                waiting_cycles /
                edge.F
            )


            # --------------------------------
            # Processing delay
            # --------------------------------

            task.processing_delay = (
                task.D /
                edge.F
            )


            # --------------------------------
            # Convert queue and processing
            # to simulation slots
            # --------------------------------

            queue_slots = math.ceil(
                task.queue_delay /
                SLOT_LENGTH
            )

            processing_slots = math.ceil(
                task.processing_delay /
                SLOT_LENGTH
            )


            # --------------------------------
            # Completion slot
            # --------------------------------

            task.E = (
                task.R +
                queue_slots +
                processing_slots
            )


            # --------------------------------
            # Total delay in seconds
            # --------------------------------

            task.total_delay = (
                task.transmission_delay +
                task.connection_delay +
                task.queue_delay +
                task.processing_delay
            )


            # --------------------------------
            # Add workload to current queue
            # --------------------------------

            current_slot_queue[
                edge_index
            ] = (
                current_queue_cycles +
                task.D
            )


    # ========================================================
    # Update edge queues
    # ========================================================

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


# ============================================================
# Collect completed tasks
# ============================================================

completed_tasks = [
    task
    for task in tasks
    if task.E != -1
]


# ============================================================
# Delay statistics
# ============================================================

number_of_completed_tasks = (
    len(completed_tasks)
)


if number_of_completed_tasks > 0:

    # --------------------------------
    # Total accumulated delays
    # --------------------------------

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


    # --------------------------------
    # Average delays
    # --------------------------------

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


    # ========================================================
    # Print results
    # ========================================================

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

