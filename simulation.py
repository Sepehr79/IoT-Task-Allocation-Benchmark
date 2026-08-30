import numpy as np
import random
import math
from scipy.stats import truncnorm
from config import *
import copy
import joblib

MODEL_FILE = "delay_prediction_pipeline.joblib"

mlp_model = None

if os.path.exists(MODEL_FILE):
    mlp_model = joblib.load(MODEL_FILE)


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
        self.virtual_task = False



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


def mlp_schedule(tasks, edges, users, slot):
    if not mlp_model:
        return
    temp_queue = [
        edge.prev_queue
        for edge in edges
    ]

    for task in tasks:

        user = users[task.userIndex]

        # ----------------------------------------------------
        # Build features
        # ----------------------------------------------------

        features = [
            task.D,
            task.B
        ]

        # Queue state
        features.extend(
            temp_queue
        )

        # Last allocation
        last_allocation = [
            0
            for _ in range(NUM_EDGES)
        ]

        if user.last_selected_edge != -1:

            last_allocation[
                user.last_selected_edge
            ] = 1

        features.extend(
            last_allocation
        )

        # Transmission rates
        features.extend(
            user.connections
        )

        # Convert to 2D array
        X = np.array(
            features,
            dtype=float
        ).reshape(1, -1)

        # ----------------------------------------------------
        # Predict delay for all edges
        # ----------------------------------------------------

        predicted_delays = mlp_model.predict(
            X
        )[0]

        # ----------------------------------------------------
        # Select Edge with minimum predicted delay
        # ----------------------------------------------------

        selected_edge = int(
            np.argmin(
                predicted_delays
            )
        )

        # ----------------------------------------------------
        # Store allocation
        # ----------------------------------------------------

        task.A = [
            0
            for _ in range(NUM_EDGES)
        ]

        task.A[selected_edge] = 1

        task.selected_edge_index = (
            selected_edge
        )

        # ----------------------------------------------------
        # Update temporary queue
        # ----------------------------------------------------

        temp_queue[selected_edge] += task.D


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


scheduling = {
    "r": random_schedule,
    "rr": round_robin_schedule,
    "g": greedy_schedule,
    "gc": greedy_context_schedule,
    "mlp": mlp_schedule
}

def calc_receive_time(task, users, edge_index):
        user = users[task.userIndex]
        task.selected_edge_index = edge_index
        transmission_rate = (
                    user.connections[edge_index]
                )
        task.data_transmission_delay = (
                    task.B /
                    transmission_rate
        )
        if (edge_index != user.last_selected_edge):
            task.context_transmission_delay = (
                user.C /
                transmission_rate
            )
            task.connection_delay = (WEBSOCKET_CONNECTION)
            if not task.virtual_task:
                user.last_selected_edge = (edge_index)
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

schedule = scheduling[SCHEDULE_ALGORITHM]

users = [
    User(USER_CONTEXT_SIZE)
    for _ in range(NUM_USERS)
]

edges = [
    Edge(1.5*10**9),
    Edge(1.5*10**9),
    Edge(2*10**9),
    Edge(2*10**9),
    Edge(2*10**9),
    Edge(2*10**9),
    Edge(2.5*10**9),
    Edge(2.5*10**9),
]

tasks = []
virs = []

server_changed = 0

if __name__ == "__main__":

    # Main simulation
    for slot in range(
        NUM_SLOTS + EXTRA_SLOTS
    ):
        print(f"Processing slot: {slot}")

        queue_before = [
            edge.prev_queue
            for edge in edges
        ]

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

                    task.queues = queue_before.copy()
                    last_edge = users[user_index].last_selected_edge
                    task.user_rates = users[user_index].connections.copy()
                    if last_edge != -1:
                        task.last_allocation[last_edge] = 1
                    tasks.append(task)


        arrived_tasks = [
            task for task in tasks
            if task.S == slot
        ]

        random.shuffle(arrived_tasks)

        if arrived_tasks:

            # Generate counterfactual samples
            if VIRTUALIZED_SLOT:

                for edge_index in range(NUM_EDGES):
                    vir_task = copy.deepcopy(arrived_tasks[0])
                    vir_task.virtual_task = True
                    calc_receive_time(
                        vir_task,
                        users,
                        edge_index
                    )
                    virs.append(vir_task)
                    tasks.append(vir_task)

            # Then perform normal allocation
            schedule(
                arrived_tasks,
                edges,
                users,
                slot
            )

        for task in arrived_tasks:
            edge_index = (task.selected_edge_index)
            user = users[task.userIndex]
            previous_edge = user.last_selected_edge
            calc_receive_time(
                task,
                users,
                edge_index
            )
            if edge_index != previous_edge:
                server_changed += 1

            
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

                if not task.virtual_task:
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
        and not task.virtual_task
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
    Virtual tasks: {
        len([task for task in tasks if task.virtual_task])
        / NUM_EDGES
    }
    {"=" * 55}
    """


    # Print to console
    print(results)


    # Append to file
    filename = (
        f"{schedule.__name__}"
        f"_{workload:.3f}_results.txt"
    )

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(results)
        file.write("\n")


    dataset = []
    i = 0
    while i < len(virs):
        task = Task(virs[i].D, virs[i].B, virs[i].S, virs[i].userIndex)
        task.queues = virs[i].queues
        task.last_allocation = virs[i].last_allocation
        task.user_rates = virs[i].user_rates
        for j in range(NUM_EDGES):
            task.delays[j] = virs[i+j].total_delay
        i += NUM_EDGES
        dataset.append(task)

    import pandas as pd

    rows = []

    for task in dataset:

        row = {
            "processing_capcacity": task.D,
            "data_size": task.B,
            "workload": workload
        }

        for i in range(NUM_EDGES):
            row[f"queue_{i}"] = task.queues[i]

        for i in range(NUM_EDGES):
            row[f"last_allocation_{i}"] = task.last_allocation[i]
        for i in range(NUM_EDGES):
            row[f"rate_{i}"] = task.user_rates[i]
        for i in range(NUM_EDGES):
            row[f"delay_{i}"] = task.delays[i]
        

        rows.append(row)

    df = pd.DataFrame(rows)

    df.to_csv(
            "allocation_dataset.csv",
            mode="a",
            header=not os.path.exists("allocation_dataset.csv"),
            index=False
        )
        


