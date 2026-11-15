"""Vehicle routing with time windows using PyVRP's Hybrid Genetic Search."""
import numpy as np
import pandas as pd
from pyvrp import Model
from pyvrp.stop import MaxRuntime

SHIFT_START_MIN = 330
HORIZON_MIN = 630
MAX_ROUTE_MIN = 480
MIN_ROUTE_MIN = 360
FIXED_TOL_MIN = 0
SERVICE_MIN = 30
TIME_LIMIT_S = 300
NUM_RUNS = 3                    # fewer, longer runs converge to shorter routes (was 6)
SEED = 42
VEHICLE_FIXED = 100_000
FLEET = {4000: 5, 3700: 2, 4800: 1, 4600: 4, 5000: 2, 2800: 1, 1500: 1}

# Selection among runs: distance now dominates (was 2.0 / 3000 / 5000 / 50).
BALANCE_PENALTY = 1.0
DURATION_BALANCE_PENALTY = 300
SHORTFALL_PENALTY = 500

loc = pd.read_csv("locations.csv", encoding="utf-8-sig")
columns = list(loc.columns)
ORG, ADDRESS, LAT, LON, DEMAND, FIXED, SMALL_ONLY = columns[1:8]
T = (np.loadtxt("time_matrix_sec.csv", delimiter=",") / 60).round().astype(int)
D = np.loadtxt("distance_matrix_m.csv", delimiter=",").round().astype(int)

# Each row is one stop; if the CSV has no service_objects column, count 1 object per stop.
if "service_objects" not in loc.columns:
    loc["service_objects"] = 1

n = len(loc)
demand = loc[DEMAND].astype(int).tolist()
service = [0] + [SERVICE_MIN * int(count) for count in loc["service_objects"].iloc[1:]]
capacities = [cap for cap, count in FLEET.items() for _ in range(count)]
print(f"Stops: {n-1}; vehicles: {len(capacities)}; demand: {sum(demand)} kg; fleet capacity: {sum(capacities)} kg.")

model = Model()
depot_location = model.add_location(x=float(loc.loc[0, LON]), y=float(loc.loc[0, LAT]))
depot = model.add_depot(location=depot_location, tw_early=0, tw_late=HORIZON_MIN)
for capacity, count in FLEET.items():
    model.add_vehicle_type(
        num_available=count, capacity=[capacity, n if capacity == 1500 else 0], start_depot=depot, end_depot=depot,
        tw_early=0, tw_late=HORIZON_MIN, shift_duration=MAX_ROUTE_MIN,
        fixed_cost=VEHICLE_FIXED,
    )

client_locations = []
for node in range(1, n):
    fixed = pd.to_numeric(pd.Series([loc.loc[node, FIXED]]), errors="coerce").iloc[0]
    early, late = 8 * 60 - SHIFT_START_MIN, HORIZON_MIN
    if pd.notna(fixed):
        early = max(0, int(fixed) - SHIFT_START_MIN)
        late = early + FIXED_TOL_MIN
    required_small_truck = int(loc.loc[node, SMALL_ONLY]) == 1
    client_location = model.add_location(x=float(loc.loc[node, LON]), y=float(loc.loc[node, LAT]))
    client_locations.append(client_location)
    model.add_client(
        location=client_location, delivery=[int(demand[node]), int(required_small_truck)], service_duration=int(service[node]),
        tw_early=early, tw_late=late, release_time=0, required=True,
    )

locations = [depot_location, *client_locations]
for i, frm in enumerate(locations):
    for j, to in enumerate(locations):
        if i != j:
            model.add_edge(frm, to, distance=int(D[i, j]), duration=int(T[i, j]))

print("Vehicle shift starts no earlier than 05:30; fixed delivery times are exact.", flush=True)
solutions = []
run_time = max(1, TIME_LIMIT_S // NUM_RUNS)
for run in range(NUM_RUNS):
    result = model.solve(stop=MaxRuntime(run_time), seed=SEED + run, display=True)
    if result.best is not None and result.best.is_feasible():
        solutions.append(result.best)
if not solutions:
    raise SystemExit("PyVRP did not find a feasible solution within the search limit.")

def balance_score(candidate):
    routes = candidate.routes()
    capacities_by_type = list(FLEET)
    load_shortfall = sum(
        max(0, int(np.ceil(0.9 * capacities_by_type[route.vehicle_type()])) - int(route.delivery()[0]))
        for route in routes
    )
    distances = np.asarray([route.distance() for route in routes], dtype=float)
    durations = np.asarray([route.duration() for route in routes], dtype=float)
    shortfall_penalty = SHORTFALL_PENALTY * sum(max(0, MIN_ROUTE_MIN - duration) for duration in durations)
    route_cost = (
        distances.sum()
        + BALANCE_PENALTY * distances.std()
        + DURATION_BALANCE_PENALTY * durations.std()
        + shortfall_penalty
    )
    return load_shortfall, float(route_cost)

solution = min(solutions, key=balance_score)
route_distances = np.asarray([route.distance() for route in solution.routes()], dtype=float)
route_durations = np.asarray([route.duration() for route in solution.routes()], dtype=float)
load_shortfall_kg, route_objective = balance_score(solution)
print(f"Selected objective: {route_objective:.0f}; load shortfall below 90%: {load_shortfall_kg} kg (distance {route_distances.sum()/1000:.1f} km, route-distance std dev {route_distances.std()/1000:.1f} km, route-duration mean {route_durations.mean():.0f} min, std dev {route_durations.std():.0f} min, range {route_durations.min():.0f}-{route_durations.max():.0f} min).")

if not solution.is_feasible():
    print(f"Best solution is infeasible: time warp={solution.time_warp()}, unplanned={solution.unplanned()}.")
    for route in solution.routes():
        print(f"Route type={route.vehicle_type()}, clients={route.num_clients()}, excess_load={route.excess_load()}, time_warp={route.time_warp()}.")
    raise SystemExit("No feasible solution found; routes.csv was not written.")

route_headers = ["vehicle", "capacity_kg", "order", "stop_id", "organization", "address", "arrival", "demand_kg"]
rows = []
total_distance = 0
served = set()
vehicle_type_offsets = {}
for route in solution.routes():
    vehicle_type = route.vehicle_type()
    capacity = list(FLEET)[vehicle_type]
    vehicle_number = sum(FLEET[cap] for cap in list(FLEET)[:vehicle_type]) + vehicle_type_offsets.get(vehicle_type, 0)
    vehicle_type_offsets[vehicle_type] = vehicle_type_offsets.get(vehicle_type, 0) + 1
    activities = [activity for activity in route.schedule() if activity.is_client()]
    nodes = [int(activity.idx) + 1 for activity in activities]
    if not nodes:
        continue
    route_distance = sum(int(D[a, b]) for a, b in zip([0, *nodes], [*nodes, 0]))
    load = sum(demand[node] for node in nodes)
    total_distance += route_distance
    stops = [(node, int(activity.start_time)) for node, activity in zip(nodes, activities)]
    start_minute = int(next(activity for activity in route.schedule() if activity.is_depot()).start_time)
    rows.append({"vehicle": vehicle_number, "capacity_kg": capacity, "order": 0, "stop_id": 0,
                 "organization": loc.loc[0, ORG], "address": loc.loc[0, ADDRESS],
                 "arrival": f"{(start_minute+SHIFT_START_MIN)//60:02d}:{(start_minute+SHIFT_START_MIN)%60:02d}",
                 "demand_kg": 0})
    served.update(nodes)
    return_minute = int(route.end_time())
    print(f"Vehicle {vehicle_number + 1} ({capacity} kg): load {load} kg ({load/capacity:.0%}), {route_distance/1000:.1f} km, return {(return_minute+SHIFT_START_MIN)//60:02d}:{(return_minute+SHIFT_START_MIN)%60:02d}")
    for order, (node, minute) in enumerate(stops, start=1):
        rows.append({"vehicle": vehicle_number, "capacity_kg": capacity, "order": order,
                     "stop_id": node, "organization": loc.loc[node, ORG], "address": loc.loc[node, ADDRESS],
                     "arrival": f"{(minute+SHIFT_START_MIN)//60:02d}:{(minute+SHIFT_START_MIN)%60:02d}",
                     "demand_kg": demand[node]})
unserved = [node for node in range(1, n) if node not in served]
if unserved:
    print(f"Unserved stops: {len(unserved)}, {sum(demand[node] for node in unserved)} kg: {unserved}")
pd.DataFrame(rows, columns=route_headers).to_csv("routes.csv", index=False, encoding="utf-8-sig")
print(f"Total distance: {total_distance/1000:.1f} km; routes saved to routes.csv")