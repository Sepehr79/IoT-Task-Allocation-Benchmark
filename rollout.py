import copy
import math
import time
import pickle
from itertools import product
from functools import partial
from types import SimpleNamespace

from simulation import *   # greedy_context_schedule, users, edges, NUM_EDGES, NUM_USERS, NUM_SLOTS,
                           # EXTRA_SLOTS, SLOT_LENGTH, USER_CONTEXT_SIZE, WEBSOCKET_CONNECTION, ...

# Base scheduling policy used for the baseline run and for every "greedy" continuation inside the rollout.
# It must have the signature (tasks, edges, users, slot) and set task.selected_edge_index.
scheduling_algorithm = greedy_context_schedule
ALGO_NAME = scheduling_algorithm.__name__

# ============================================================
# Parameters
# ============================================================
ROLLOUT_DEPTHS = (0, 1)      # depth = number of levels (one level per slot) whose leaves are fully enumerated
                             # 0 = plain base policy from the root (no branching)
                             # 1 = all assignments of the current slot, then base policy from every leaf
                             # 2 = all assignments of the current and the next slot, then base policy from every leaf
ROLLOUT_K = 20               # after the depth: how many slots to continue with the base policy (None = to the horizon)
ADAPTIVE_LEAF_LIMIT = None   # if the estimated number of leaves exceeds this, the last levels are dropped (None = never)
TERMINAL_WEIGHT = 0.0        # optional penalty on the queue left at the end of the horizon

TEST_SLOTS = 200             # run only the first slots (None = whole simulation)

PROGRESS_EVERY_SLOTS = 100   # print a progress line every N slots (0 = disabled)
PROGRESS_EVERY_SECONDS = 5   # if nothing was printed for this long, print an in-search progress line (0 = disabled)
PRINT_DECISIONS = True       # per decision: slot, depth, tasks per level, leaves per level, leaves at depth

F = [edge.F for edge in edges]
SERVICE = [edge.service_rate for edge in edges]

STATS = {"decisions": 0, "nodes": 0, "leaves": 0, "reduced": 0,
         "max_est": 0, "max_frontier": 0, "levels": {}}
SLOT_TASKS = {}
_LAST_PRINT = [time.perf_counter()]


# ============================================================
# Lightweight simulation state
# ============================================================
class State:
    def __init__(self):
        self.queues = [0.0] * NUM_EDGES      # queued cycles per edge
        self.last_edge = [-1] * NUM_USERS    # last edge selected by each user
        self.pending = {}                    # reception slot R -> [(edge, D, comm_delay)]
        self.pend_cp = 0.0                   # sum of (comm + processing) of in-flight tasks

    def copy(self):
        s = State.__new__(State)
        s.queues = self.queues.copy()
        s.last_edge = self.last_edge.copy()
        s.pending = {k: v.copy() for k, v in self.pending.items()}
        s.pend_cp = self.pend_cp
        return s


def comm_delay(task, e, last_edge):
    """Data transmission + context transmission + connection delay (same as calc_receive_time)."""
    rate = task.user_rates[e]
    c = task.B / rate
    if last_edge[task.userIndex] != e:
        c += USER_CONTEXT_SIZE / rate + WEBSOCKET_CONNECTION
    return c


def assign(state, task, e):
    """Commit task -> edge e on a (hypothetical) state; the task arrives at slot R."""
    c = comm_delay(task, e, state.last_edge)
    R = task.S + math.ceil(c / SLOT_LENGTH)
    state.pending.setdefault(R, []).append((e, task.D, c))
    state.pend_cp += c + task.D / F[e]
    state.last_edge[task.userIndex] = e


def process_slot(state, slot):
    """Process the tasks arriving at `slot`; returns (total, n, comm, queue, proc) delays."""
    arrivals = state.pending.pop(slot, [])
    qb = state.queues
    cur = [0.0] * NUM_EDGES
    comm_s = queue_s = proc_s = 0.0

    for e, D, c in arrivals:
        comm_s += c
        queue_s += (qb[e] + cur[e]) / F[e]
        proc_s += D / F[e]
        cur[e] += D

    state.queues = [
        max(qb[e] - SERVICE[e], 0.0) + cur[e]
        for e in range(NUM_EDGES)
    ]
    state.pend_cp -= comm_s + proc_s
    return (comm_s + queue_s + proc_s, len(arrivals), comm_s, queue_s, proc_s)


# ============================================================
# Base policy (scheduling_algorithm from simulation.py)
# ============================================================
# The scheduler from simulation.py reads edge.prev_queue and user.last_selected_edge, i.e. the real
# simulation state. To evaluate it on hypothetical states, only these two fields are set on
# lightweight proxy objects, so the real `edges` and `users` are never modified.
_P_EDGES = [SimpleNamespace(F=e.F, prev_queue=0.0) for e in edges]
_P_USERS = [SimpleNamespace(C=u.C, last_selected_edge=-1) for u in users]


def base_assignment(state, tasks):
    """Base-policy assignment for one slot, evaluated on `state` (state is not modified)."""
    if not tasks:
        return ()
    for i in range(NUM_EDGES):
        _P_EDGES[i].prev_queue = state.queues[i]
    for t in tasks:
        _P_USERS[t.userIndex].last_selected_edge = state.last_edge[t.userIndex]
    scheduling_algorithm(tasks, _P_EDGES, _P_USERS, None)
    return tuple(t.selected_edge_index for t in tasks)


def base_assign_slot(state, tasks):
    for t, b in zip(tasks, base_assignment(state, tasks)):
        assign(state, t, b)


def base_schedule(state, tasks, slot, slot_tasks, depth=None):
    """Plain base policy for the real simulation (no rollout)."""
    for t, b in zip(tasks, base_assignment(state, tasks)):
        t._changed = state.last_edge[t.userIndex] != b
        assign(state, t, b)


# ============================================================
# Exact slot-level rollout with level-by-level search (BFS)
#   Each level = one slot; branches = all possible assignments of that slot's tasks (NUM_EDGES^n).
#   BFS: first all level-0 nodes are built, then all of their children (level 1), and so on.
#   No pruning: every node of a level goes to the next level (the frontier).
#   At the last level, the base policy continues from every leaf for k slots.
#   The root branch of the best leaf is applied to the real state, and the procedure repeats next slot.
# ============================================================
class Search:
    def __init__(self, tail_end):
        self.best = float("inf")
        self.best_assign = None    # root assignment of the best leaf
        self.tail_end = tail_end
        self.nodes = 0
        self.leaves = 0
        self.max_frontier = 0
        self.level_stats = []      # per-level statistics


def all_assignments(state, tasks):
    """All NUM_EDGES^n assignments of one slot; the first one yielded is the base-policy assignment."""
    n = len(tasks)
    if n == 0:
        yield ()
        return
    g = base_assignment(state, tasks)
    yield g
    for combo in product(range(NUM_EDGES), repeat=n):
        if combo != g:
            yield combo


def base_tail(state, last_slot, tail_end):
    """After the last level: base policy on the real tasks up to tail_end, then drain in-flight tasks."""
    total = 0.0
    last = last_slot
    for f in range(last_slot + 1, tail_end):
        base_assign_slot(state, SLOT_TASKS.get(f, []))
        total += process_slot(state, f)[0]
        last = f
    if state.pending:
        for f in range(last + 1, max(state.pending) + 1):
            total += process_slot(state, f)[0]
    if TERMINAL_WEIGHT:
        total += TERMINAL_WEIGHT * sum(state.queues[i] / F[i] for i in range(NUM_EDGES))
    return total


def estimate_leaves(levels):
    est = 1
    for _, tasks in levels:
        est *= NUM_EDGES ** len(tasks)
    return est


def _progress(ctx, li, n_levels, parent_idx, frontier_size, generated, level_t0):
    """Print an in-search progress line every PROGRESS_EVERY_SECONDS seconds."""
    if not PROGRESS_EVERY_SECONDS:
        return
    now = time.perf_counter()
    if now - _LAST_PRINT[0] < PROGRESS_EVERY_SECONDS:
        return
    done = parent_idx / frontier_size if frontier_size else 1.0
    el = now - level_t0
    eta = el / done - el if done > 0 else float("nan")
    print(f"    ... BFS level {li + 1}/{n_levels} | parents {parent_idx:,}/{frontier_size:,} "
          f"({done:.1%}) | generated={generated:,} | leaves={ctx.leaves:,} | "
          f"best={ctx.best:.4f} | level ETA={eta:.0f}s", flush=True)
    _LAST_PRINT[0] = now


def bfs(ctx, state, levels):
    """
    levels   = [(slot, [tasks]), ...]
    frontier = list of (state, acc, root_assignment); acc = exact cost of the slots processed so far.
    """
    n_levels = len(levels)
    frontier = [(state, 0.0, None)]

    for li, (slot, tasks) in enumerate(levels):
        last_level = (li == n_levels - 1)
        next_frontier = []
        generated = 0
        frontier_in = len(frontier)
        t0 = time.perf_counter()

        for parent_idx, (st, acc, root) in enumerate(frontier):
            for choice in all_assignments(st, tasks):
                ctx.nodes += 1
                generated += 1
                _progress(ctx, li, n_levels, parent_idx, frontier_in, generated, t0)

                s = st.copy()
                for t, e in zip(tasks, choice):
                    assign(s, t, e)
                new_root = choice if li == 0 else root

                acc2 = acc + process_slot(s, slot)[0]

                if last_level:
                    # leaf: continue with the base policy and evaluate the total cost
                    ctx.leaves += 1
                    total = acc2 + base_tail(s, slot, ctx.tail_end)
                    if total < ctx.best:
                        ctx.best, ctx.best_assign = total, new_root
                else:
                    next_frontier.append((s, acc2, new_root))

        ctx.level_stats.append(dict(level=li, slot=slot, n_tasks=len(tasks),
                                    frontier_in=frontier_in, generated=generated,
                                    kept=len(next_frontier),
                                    sec=time.perf_counter() - t0))
        ctx.max_frontier = max(ctx.max_frontier, len(next_frontier), frontier_in)

        frontier = next_frontier
        if not frontier:
            break


def rollout_schedule(state, tasks, slot, slot_tasks, depth, k=ROLLOUT_K):
    global SLOT_TASKS
    SLOT_TASKS = slot_tasks

    # depth 0: plain base policy (identical to base_schedule)
    if depth is not None and depth <= 0:
        STATS["decisions"] += 1
        for t, e in zip(tasks, base_assignment(state, tasks)):
            t._changed = state.last_edge[t.userIndex] != e
            assign(state, t, e)
        return

    # levels: current slot + (depth - 1) following slots; one level per slot
    depth_end = NUM_SLOTS if depth is None else min(NUM_SLOTS, slot + depth)

    levels = [(slot, tasks)]
    for f in range(slot + 1, depth_end):
        levels.append((f, slot_tasks.get(f, [])))

    # optional: if the tree is too large, drop the last levels
    est = estimate_leaves(levels)
    STATS["max_est"] = max(STATS["max_est"], est)
    if ADAPTIVE_LEAF_LIMIT is not None:
        while len(levels) > 1 and estimate_leaves(levels) > ADAPTIVE_LEAF_LIMIT:
            levels.pop()
            STATS["reduced"] += 1

    last_slot = levels[-1][0]
    tail_end = NUM_SLOTS if k is None else min(NUM_SLOTS, last_slot + 1 + k)

    if PRINT_DECISIONS:
        # nodes per level = product of NUM_EDGES^(tasks) up to that level; last level = leaves at depth
        counts, cum = [], 1
        for _, tk in levels:
            cum *= NUM_EDGES ** len(tk)
            counts.append(cum)
        tasks_txt = ", ".join(f"L{i + 1}={len(tk)}" for i, (_, tk) in enumerate(levels))
        print(f"    [slot {slot}] depth={len(levels)} | base policy: {ALGO_NAME} "
              f"| tasks per level: {tasks_txt} "
              f"| leaves at depth (base policy continues): {counts[-1]:,}", flush=True)

    ctx = Search(tail_end)
    bfs(ctx, state, levels)

    best = ctx.best_assign if ctx.best_assign is not None else base_assignment(state, tasks)

    STATS["decisions"] += 1
    STATS["nodes"] += ctx.nodes
    STATS["leaves"] += ctx.leaves
    STATS["max_frontier"] = max(STATS["max_frontier"], ctx.max_frontier)
    for ls in ctx.level_stats:
        agg = STATS["levels"].setdefault(
            ls["level"], dict(generated=0, kept=0, sec=0.0, count=0))
        agg["generated"] += ls["generated"]
        agg["kept"] += ls["kept"]
        agg["sec"] += ls["sec"]
        agg["count"] += 1

    now = time.perf_counter()
    if PROGRESS_EVERY_SECONDS and now - _LAST_PRINT[0] >= PROGRESS_EVERY_SECONDS:
        print(f"    ... slot {slot} decided | decisions={STATS['decisions']:,}", flush=True)
        _LAST_PRINT[0] = now

    # apply the root branch of the winning leaf to the real state (whole slot at once)
    for t, e in zip(tasks, best):
        t._changed = state.last_edge[t.userIndex] != e
        assign(state, t, e)


# ============================================================
# Full simulation
# ============================================================
def run_simulation(slot_tasks, policy, depth=None, label="", quiet=False):
    state = State()
    n_done = 0
    sums = dict(total=0.0, comm=0.0, queue=0.0, proc=0.0)
    server_changed = 0
    sched_time = 0.0

    STATS.update(decisions=0, nodes=0, leaves=0, reduced=0,
                 max_est=0, max_frontier=0, levels={})
    wall_start = time.perf_counter()
    _LAST_PRINT[0] = wall_start
    total_slots = (TEST_SLOTS or NUM_SLOTS) + EXTRA_SLOTS

    for slot in range(total_slots):

        arrived = slot_tasks.get(slot, []) if slot < NUM_SLOTS else []
        arrived = [copy.copy(t) for t in arrived]    # do not modify the original data

        if arrived:
            t0 = time.perf_counter()
            policy(state, arrived, slot, slot_tasks, depth)
            sched_time += time.perf_counter() - t0
            server_changed += sum(1 for t in arrived if t._changed)

        total, n, comm_s, queue_s, proc_s = process_slot(state, slot)
        n_done += n
        sums["total"] += total
        sums["comm"] += comm_s
        sums["queue"] += queue_s
        sums["proc"] += proc_s

        if not quiet and PROGRESS_EVERY_SLOTS and (slot + 1) % PROGRESS_EVERY_SLOTS == 0:
            elapsed = time.perf_counter() - wall_start
            frac = (slot + 1) / total_slots
            eta = elapsed / frac - elapsed
            print(f"[{label}] slot {slot + 1}/{total_slots} ({frac:6.1%}) | "
                  f"decisions={STATS['decisions']:,} | leaves={STATS['leaves']:,} | "
                  f"max_frontier={STATS['max_frontier']:,} | "
                  f"elapsed={elapsed:7.1f}s | ETA={eta:7.1f}s", flush=True)
            _LAST_PRINT[0] = time.perf_counter()

    n_done = max(n_done, 1)
    return {
        "completed": n_done,
        "avg_total": sums["total"] / n_done,
        "avg_comm": sums["comm"] / n_done,
        "avg_queue": sums["queue"] / n_done,
        "avg_proc": sums["proc"] / n_done,
        "server_changed": server_changed,
        "sched_time_s": sched_time,
        "decisions": STATS["decisions"],
        "avg_nodes": STATS["nodes"] / max(STATS["decisions"], 1),
        "avg_leaves": STATS["leaves"] / max(STATS["decisions"], 1),
        "reduced": STATS["reduced"],
        "max_est": STATS["max_est"],
        "max_frontier": STATS["max_frontier"],
        "levels": {k: v.copy() for k, v in STATS["levels"].items()},
    }


def print_result(name, r):
    extra = ""
    if r.get("decisions"):
        extra = (f"Decisions (slots): {r['decisions']:,} | "
                 f"avg nodes: {r['avg_nodes']:,.0f} | avg leaves: {r['avg_leaves']:,.0f}\n"
                 f"Max estimated leaves: {r['max_est']:,} | "
                 f"decisions with reduced depth: {r['reduced']:,}\n"
                 f"Max BFS frontier size: {r['max_frontier']:,}\n"
                 f"Avg time/decision: {1000 * r['sched_time_s'] / r['decisions']:.1f} ms\n")
        if r.get("levels"):
            extra += ("\nBFS per-level summary (averaged over decisions):\n"
                      f"{'level':>5} | {'generated':>14} | {'kept (frontier)':>15} | "
                      f"{'time(ms)':>9}\n")
            for li in sorted(r["levels"]):
                a = r["levels"][li]
                c = max(a["count"], 1)
                extra += (f"{li:>5} | {a['generated'] / c:>14,.1f} | {a['kept'] / c:>15,.1f} | "
                          f"{1000 * a['sec'] / c:>9.2f}\n")
    print(f"""
{'=' * 55}
Algorithm: {name}
Base policy: {ALGO_NAME}
Completed tasks: {r['completed']}
Average transmission+connection delay: {r['avg_comm']:.4f} s
Average queue delay: {r['avg_queue']:.4f} s
Average processing delay: {r['avg_proc']:.4f} s
Average total delay: {r['avg_total']:.4f} s
Server changes: {r['server_changed']}
Scheduling time: {r['sched_time_s']:.2f} s
{extra}{'=' * 55}""")


if __name__ == "__main__":
    with open("slot_tasks.pkl", "rb") as file:
        slot_tasks = pickle.load(file)

    print(f"NUM_SLOTS={NUM_SLOTS}, EXTRA_SLOTS={EXTRA_SLOTS}, TEST_SLOTS={TEST_SLOTS}")
    print(f"slots in pickle: {len(slot_tasks)}, "
          f"tasks in pickle: {sum(len(v) for v in slot_tasks.values())}")

    if TEST_SLOTS:
        slot_tasks = {s: v for s, v in slot_tasks.items() if s < TEST_SLOTS}
        print(f">>> TEST MODE: only the first {TEST_SLOTS} slots")

    print(f">>> Running base policy ({ALGO_NAME})")
    r_base = run_simulation(slot_tasks, base_schedule, label="base", quiet=True)
    print_result(f"{ALGO_NAME} (base)", r_base)

    for d in ROLLOUT_DEPTHS:
        print(f"\n>>> Starting exact slot-level rollout (BFS): depth={d}, k={ROLLOUT_K}, "
              f"base_policy={ALGO_NAME}, adaptive_limit={ADAPTIVE_LEAF_LIMIT}")
        r = run_simulation(
            slot_tasks,
            partial(rollout_schedule, k=ROLLOUT_K),
            depth=d,
            label=f"rollout-BFS d={d}",
        )
        print_result(f"exact slot-level rollout BFS (depth={d}, then base policy {ALGO_NAME} for k={ROLLOUT_K})", r)