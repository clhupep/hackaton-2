"""Vehicle routing with time windows. Run: python vrptw.py"""
import numpy as np
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

SHIFT_START_MIN = 5 * 60 + 30
HORIZON_MIN = 10 * 60 + 30
MAX_ROUTE_MIN = 8 * 60
FIXED_TOL_MIN = 0
SERVICE_MIN = 30
SERVICE_KG_STEP = 200
MIN_LOAD_SHARE = 0.9
LOAD_PENALTY = 50
VEHICLE_FIXED = 100000
TIME_LIMIT_S = 300
MERGE_SAME_ADDRESS = False
DROP_PENALTY = 10_000_000
FLEET = {4000: 5, 3700: 2, 4800: 1, 4600: 4, 5000: 2, 2800: 1, 1500: 1}

loc = pd.read_csv("locations_agg.csv", encoding="utf-8-sig")
columns = list(loc.columns)
ORG, ADDRESS, LAT, LON, DEMAND, FIXED, SMALL_ONLY = columns[1:8]
T = (np.loadtxt("time_matrix_sec.csv", delimiter=",") / 60).round().astype(int)
D = np.loadtxt("distance_matrix_m.csv", delimiter=",").round().astype(int)

if "service_objects" not in loc.columns:
    loc["service_objects"] = 1

if MERGE_SAME_ADDRESS:
    grouped = loc.groupby([LAT, LON], sort=False)
    merged = []
    for _, sub in grouped:
        if 0 in sub.index:
            continue
        fixed_times = pd.to_numeric(sub[FIXED], errors="coerce")
        merged.append({
            columns[0]: int(sub.iloc[0, 0]),
            ORG: " | ".join(sub[ORG].astype(str)),
            ADDRESS: sub[ADDRESS].iloc[0],
            LAT: sub[LAT].iloc[0],
            LON: sub[LON].iloc[0],
            DEMAND: int(sub[DEMAND].sum()),
            FIXED: fixed_times.min() if fixed_times.notna().any() else np.nan,
            SMALL_ONLY: int(sub[SMALL_ONLY].max()),
            "service_objects": len(sub),
        })
    loc = pd.concat([loc.iloc[[0]], pd.DataFrame(merged)], ignore_index=True)
    representatives = [0] + [int(row[columns[0]]) for row in merged]
    T = T[np.ix_(representatives, representatives)]
    D = D[np.ix_(representatives, representatives)]
    print(f"Merged to {len(loc)-1} stops by coordinates.")

n = len(loc)
demand = loc[DEMAND].astype(int).tolist()
service = [0] + [SERVICE_MIN * int(count) for count in loc["service_objects"].iloc[1:]]
capacities = [capacity for capacity, count in sorted(FLEET.items(), reverse=True) for _ in range(count)]
vehicle_count = len(capacities)
print(f"Stops: {n-1}; vehicles: {vehicle_count}; demand: {sum(demand)} kg; fleet capacity: {sum(capacities)} kg.")

manager = pywrapcp.RoutingIndexManager(n, vehicle_count, 0)
routing = pywrapcp.RoutingModel(manager)
distance_callback = routing.RegisterTransitCallback(lambda i, j: int(D[manager.IndexToNode(i)][manager.IndexToNode(j)]))
routing.SetArcCostEvaluatorOfAllVehicles(distance_callback)
routing.SetFixedCostOfAllVehicles(VEHICLE_FIXED)
time_callback = routing.RegisterTransitCallback(lambda i, j: int(T[manager.IndexToNode(i)][manager.IndexToNode(j)] + service[manager.IndexToNode(i)]))
routing.AddDimension(time_callback, HORIZON_MIN, HORIZON_MIN, False, "Time")
time_dimension = routing.GetDimensionOrDie("Time")
for node, row in loc.iterrows():
    if node == 0:
        continue
    earliest = 0
    latest = HORIZON_MIN
    if pd.notna(row[FIXED]):
        earliest = max(0, int(row[FIXED]) - SHIFT_START_MIN)
        latest = earliest + FIXED_TOL_MIN
    time_dimension.CumulVar(manager.NodeToIndex(node)).SetRange(earliest, latest)

for vehicle in range(vehicle_count):
    start, end = routing.Start(vehicle), routing.End(vehicle)
    time_dimension.CumulVar(start).SetRange(0, HORIZON_MIN)
    routing.solver().Add(time_dimension.CumulVar(end) - time_dimension.CumulVar(start) <= MAX_ROUTE_MIN)
    routing.AddVariableMaximizedByFinalizer(time_dimension.CumulVar(start))
    routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(end))

demand_callback = routing.RegisterUnaryTransitCallback(lambda i: demand[manager.IndexToNode(i)])
routing.AddDimensionWithVehicleCapacity(demand_callback, 0, capacities, True, "Load")
load_dimension = routing.GetDimensionOrDie("Load")
for vehicle in range(vehicle_count):
    load_dimension.SetCumulVarSoftLowerBound(routing.End(vehicle), int(MIN_LOAD_SHARE * capacities[vehicle]), LOAD_PENALTY)
for node in range(1, n):
    routing.AddDisjunction([manager.NodeToIndex(node)], DROP_PENALTY)
small_vehicles = [v for v, cap in enumerate(capacities) if cap == 1500]
for node in loc.index[loc[SMALL_ONLY] == 1]:
    routing.VehicleVar(manager.NodeToIndex(int(node))).SetValues(small_vehicles)

parameters = pywrapcp.DefaultRoutingSearchParameters()
parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
parameters.time_limit.seconds = TIME_LIMIT_S
print("Vehicle shift starts no earlier than 05:30; fixed delivery times are exact.", flush=True)
solution = routing.SolveWithParameters(parameters)
if not solution:
    raise SystemExit("No solution found within the configured search limit.")

route_headers = list(pd.read_csv("routes.csv", encoding="utf-8-sig", nrows=0).columns)
rows = []
total_distance = 0
for vehicle in range(vehicle_count):
    index = routing.Start(vehicle)
    if routing.IsEnd(solution.Value(routing.NextVar(index))):
        continue
    load = 0
    route_distance = 0
    stops = []
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        load += demand[node]
        minute = solution.Min(time_dimension.CumulVar(index)) + SHIFT_START_MIN
        stops.append((node, f"{minute//60:02d}:{minute%60:02d}"))
        following = solution.Value(routing.NextVar(index))
        route_distance += D[node][manager.IndexToNode(following)]
        index = following
    return_minute = solution.Min(time_dimension.CumulVar(index)) + SHIFT_START_MIN
    total_distance += route_distance
    print(f"Vehicle {vehicle+1} ({capacities[vehicle]} kg): load {load} kg ({load/capacities[vehicle]:.0%}), {route_distance/1000:.1f} km, return {return_minute//60:02d}:{return_minute%60:02d}")
    for order, (node, arrival) in enumerate(stops):
        rows.append({route_headers[0]: vehicle, route_headers[1]: capacities[vehicle], route_headers[2]: order,
                     route_headers[3]: node, route_headers[4]: loc.loc[node, ORG], route_headers[5]: loc.loc[node, ADDRESS],
                     route_headers[6]: arrival, route_headers[7]: demand[node]})
unserved = [node for node in range(1, n) if solution.Value(routing.NextVar(manager.NodeToIndex(node))) == manager.NodeToIndex(node)]
if unserved:
    print(f"Unserved stops: {len(unserved)}, {sum(demand[node] for node in unserved)} kg: {unserved}")
pd.DataFrame(rows, columns=route_headers).to_csv("routes.csv", index=False, encoding="utf-8-sig")
print(f"Total distance: {total_distance/1000:.1f} km; routes saved to routes.csv")
