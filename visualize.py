"""Build an offline dashboard and audit the saved routes: python visualize.py."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def minutes(value):
    hours, minute = str(value).split(":")
    return int(hours) * 60 + int(minute)


def build(folder):
    locations = pd.read_csv(folder / "locations_agg.csv", encoding="utf-8-sig")
    routes = pd.read_csv(folder / "routes.csv", encoding="utf-8-sig")
    grouped = "формат" in routes.columns and len(routes) > 0 and bool((routes["формат"] == 3).all())
    strict = "формат" in routes.columns and len(routes) > 0 and bool((routes["формат"].isin([2, 3])).all())
    if strict:
        routes = routes[routes["тип"] != "возврат"]
    travel = np.loadtxt(folder / "time_matrix_sec.csv", delimiter=",") / 60
    if strict:
        travel = np.rint(travel)
    distance = np.loadtxt(folder / "distance_matrix_m.csv", delimiter=",")
    if travel.shape != (len(locations), len(locations)) or distance.shape != travel.shape:
        raise ValueError("Matrix dimensions do not match locations.csv")
    if not np.isfinite(travel).all() or not np.isfinite(distance).all():
        raise ValueError("Matrices contain missing or infinite values")
    columns = list(routes.columns)
    vehicle, capacity, order, node, company, address, arrival, weight = columns[:8]
    groups = [locations.iloc[[0]]] + [part for _, part in locations.groupby(["lat", "lon"], sort=False) if 0 not in part.index]
    if strict:
        groups = [locations.iloc[[index]] for index in range(len(locations))]
        index_by_id = {int(row.id): index for index, row in locations.iterrows()}
    if grouped:
        groups = address_groups(locations)
        index_by_id = {int(group.iloc[0].id): index for index, group in enumerate(groups)}
    warnings = []
    results = []
    visited = set()
    for vehicle_id, route in routes.groupby(vehicle, sort=True):
        route = route.sort_values(order)
        stops = []
        route_warnings = []
        for _, row in route.iterrows():
            node_id = index_by_id[int(row[node])] if strict else int(row[node])
            if node_id < 0 or node_id >= len(groups):
                raise ValueError(f"Unknown legacy grouped node ID: {node_id}")
            members = groups[node_id]
            original = int(members.index[0])
            if str(row[address]).strip() != str(members.iloc[0, 2]).strip() or int(row[weight]) != int(members.demand_kg.sum()):
                raise ValueError("routes.csv does not match the legacy grouped locations; regenerate matching inputs")
            if node_id:
                duplicates = visited.intersection(members.index)
                if duplicates:
                    route_warnings.append(f"Повторная доставка: {sorted(duplicates)}")
                visited.update(members.index)
            service = 30 * len(members) if node_id else 0
            start = minutes(row[arrival])
            windows = [int(value) for value in members.fixed_time_min.dropna()]
            fixed_warnings = []
            if grouped and not windows and node_id and not DEFAULT_OPEN_MIN <= start <= DEFAULT_CLOSE_MIN:
                fixed_warnings.append("Вне окна 08:00–14:00")
            if any(not target <= start <= target + 30 for target in windows):
                fixed_warnings.append("Нарушено даже допущение окна [T, T+30]")
            if any(start != target for target in windows):
                fixed_warnings.append("Время отличается от заданного в Excel; допуск не согласован")
            if int(members.only_1500kg.max()) and int(row[capacity]) != 1500:
                fixed_warnings.append("Центр: требуется машина 1500 кг")
            route_warnings.extend(f"{row[address]}: {message}" for message in fixed_warnings)
            stops.append(dict(id=node_id, original=original, members=[int(value) for value in members.id], name=str(row[company]), address=str(row[address]), lat=float(members.iloc[0].lat), lon=float(members.iloc[0].lon), kg=int(row[weight]), time=start, service=service, objects=len(members) if node_id else 0, windows=windows, issues=fixed_warnings))
        if not stops or stops[0]["id"] != 0:
            raise ValueError("Each saved route must start at the depot")
        kilometers = 0.0
        earliest = float(stops[0]["time"])
        model_earliest = earliest
        for previous, stop in zip(stops, stops[1:]):
            leg = float(round(travel[previous["original"], stop["original"]]))
            kilometers += float(distance[previous["original"], stop["original"]]) / 1000
            required = previous["time"] + previous["service"] + leg
            if stop["time"] + 1e-6 < required:
                stop["issues"].append(f"Недостаточно времени на разгрузку/дорогу: {required - stop['time']:.1f} мин")
            earliest = max(earliest + previous["service"] + leg, stop["time"])
            model_earliest = max(model_earliest + (30 * previous["objects"] if previous["id"] else 0) + round(leg), stop["time"])
        last = stops[-1]
        back = float(round(travel[last["original"], 0]))
        kilometers += float(distance[last["original"], 0]) / 1000
        end = earliest + last["service"] + back
        model_end = model_earliest + (30 * last["objects"] if last["id"] else 0) + round(back)
        duration = end - stops[0]["time"]
        if grouped and end > DEPOT_CLOSE_MIN:
            route_warnings.append("Возврат после 18:00")
        if duration > 480:
            route_warnings.append(f"Смена с учётом разгрузок: минимум {duration / 60:.2f} ч > 8 ч")
        load = sum(stop["kg"] for stop in stops)
        cap = int(route.iloc[0][capacity])
        if not 0.9 * cap <= load <= cap:
            route_warnings.append("Загрузка вне обязательных 90–100%")
        if any(stop["issues"] for stop in stops):
            route_warnings.append("Есть замечания к расписанию остановок")
        results.append(dict(vehicle=int(vehicle_id), capacity=cap, kg=load, load=round(100 * load / cap, 1), km=round(kilometers, 2), start=stops[0]["time"], end=round(end, 1), model_end=model_end, duration=round(duration, 1), stops=stops, issues=route_warnings))
    unserved = locations.loc[~locations.index.isin(visited) & (locations.index != 0)]
    warnings.extend([
        "Сохранённый план объединяет объекты по координатам и отводит 30 минут на адрес. Проверка ниже считает 30 минут на каждый торговый объект, как сказано в транскрипции.",
        "Excel задаёт время выгрузки, но не разрешает опоздание на 30 минут. Окна [T, T+30] — допущение исходного решателя.",
        "Транскрипция называет рабочее время 08:00–17:00, Excel содержит доставки в 06:00 и 07:00 и смену 8 часов. Начало смены требует согласования.",
        "Координаты депо и клиентов не подтверждены источниками геокодирования. В PDF адрес загрузки: Господарчая 28/30; его нужно сверить с координатами депо.",
        "Расстояния рассчитаны по имеющейся матрице OSRM car; это не проверка грузовых ограничений и не учёт пробок.",
        "Линии на схеме соединяют остановки напрямую; это порядок посещения, а не геометрия автомобильных дорог.",
        "Нет исходных действующих маршрутов для сравнения экономии; уменьшение пробега относительно текущей работы не доказано.",
        "Нижняя оценка числа машин: 16. Пятнадцать самых вместительных машин перевозят 63 400 кг, а спрос составляет 63 630 кг.",
    ])
    if strict:
        warnings = ["Каждый объект обслуживается отдельно 30 минут; точное начало выгрузки; все заказы обязательны; загрузка 90–100%; смена до 8 часов.", "Время дороги округляется до ближайшей минуты.", "Линии показывают порядок остановок, без геометрии дорог. Координаты депо требуют подтверждения.", "Результат Excel: result.xlsx. Ранние выезды являются допущением; см. лист условий."]
    if grouped:
        warnings = ["30 минут на адрес для всех организаций; точное время любой организации действует на адрес; иначе окно 08:00–14:00. Прибытие в 14:00 разрешено, разгрузка может закончиться позже.", "Все заказы обязательны; загрузка 90–100%; смена до 8 часов; возврат до 18:00.", "Время дороги округляется до ближайшей минуты; геометрия дорог на схеме не показана.", "Результат Excel: result.xlsx. Координаты депо и ранний выезд требуют подтверждения."]
    payload = dict(routes=results, warnings=warnings, total_kg=int(locations.demand_kg.sum()), delivered_kg=sum(route["kg"] for route in results), total_objects=len(locations)-1, served_objects=len(visited), unserved=[dict(id=int(row.id), name=str(row.iloc[1]), lat=float(row.lat), lon=float(row.lon), kg=int(row.demand_kg)) for _, row in unserved.iterrows()], total_km=round(sum(route["km"] for route in results), 1))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    status_path = args.folder / "solver_status.json"
    if status_path.exists() and not json.loads(status_path.read_text(encoding="utf-8"))["feasible"]:
        print("Допустимый план не найден; см. result.xlsx и dashboard.html")
        return
    payload = build(args.folder)
    template = (args.folder / "dashboard_template.html").read_text(encoding="utf-8")
    data = json.dumps(payload, ensure_ascii=False).replace("<", "\u003c")
    if status_path.exists():
        excel_file = json.loads(status_path.read_text(encoding="utf-8")).get("excel_file", "result.xlsx")
        template = template.replace("result.xlsx", excel_file)
    (args.folder / "dashboard.html").write_text(template.replace("__DATA__", data), encoding="utf-8")
    (args.folder / "audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ["total_kg", "delivered_kg", "total_objects", "served_objects", "total_km"]}, ensure_ascii=False))
    for route in payload["routes"]:
        print(f"Vehicle {route['vehicle']}: {route['load']}%, {route['km']} km, >= {route['duration']} min; issues: {len(route['issues'])}")
    print("Saved dashboard.html and audit.json")


if __name__ == "__main__":
    main()


