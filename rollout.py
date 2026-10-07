import pickle
from simulation import *

if __name__ == "__main__":
    with open("slot_tasks.pkl", "rb") as file:
        slot_tasks = pickle.load(file)

    print(len(slot_tasks))