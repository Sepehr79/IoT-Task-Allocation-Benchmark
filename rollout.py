import pickle
from simulation import *


def exhaustive_schedule(
    tasks,
    edges,
    users,
    slot
):

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

    for task_index, task in enumerate(tasks):

        edge_index = best_solution[task_index]

        task.A = [0] * num_edges

        task.A[edge_index] = 1

        task.selected_edge_index = edge_index

    return best_delay

if __name__ == "__main__":
    with open("slot_tasks.pkl", "rb") as file:
        slot_tasks = pickle.load(file)

    print(len(slot_tasks))