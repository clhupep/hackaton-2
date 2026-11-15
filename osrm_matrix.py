"""Матрицы времени (сек) и расстояний (м) через OSRM Table API.
Запуск:  python osrm_matrix.py [--url http://localhost:5000]
По умолчанию — публичный демо-сервер (только для небольших объёмов, ~16 запросов нам хватит).
Для своего сервера: Docker + belarus-latest.osm.pbf с Geofabrik, профиль car.
"""
import argparse, time, requests, numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="https://router.project-osrm.org")
ap.add_argument("--block", type=int, default=45)   # источники+назначения в запросе <= 100 (лимит демо-сервера)
a = ap.parse_args()

loc = pd.read_csv("locations_agg.csv", encoding="utf-8-sig")
coords = list(zip(loc["lon"], loc["lat"]))          # OSRM: долгота,широта
n = len(coords)
dur = np.zeros((n, n)); dist = np.zeros((n, n))

for i0 in range(0, n, a.block):
    src = list(range(i0, min(i0 + a.block, n)))
    for j0 in range(0, n, a.block):
        dst = list(range(j0, min(j0 + a.block, n)))
        idx = src + [j for j in dst if j not in src]
        pos = {g: k for k, g in enumerate(idx)}
        cs = ";".join(f"{coords[g][0]},{coords[g][1]}" for g in idx)
        r = requests.get(f"{a.url}/table/v1/driving/{cs}",
                         params={"sources": ";".join(str(pos[g]) for g in src),
                                 "destinations": ";".join(str(pos[g]) for g in dst),
                                 "annotations": "duration,distance"}, timeout=60)
        r.raise_for_status(); j = r.json()
        assert j["code"] == "Ok", j
        dur[np.ix_(src, dst)] = np.array(j["durations"], dtype=float)
        dist[np.ix_(src, dst)] = np.array(j["distances"], dtype=float)
        time.sleep(1)
        print(f"блок {i0}-{src[-1]} x {j0}-{dst[-1]} готов")

if np.isnan(dur).any(): print("ВНИМАНИЕ: есть точки, не привязанные к дорогам (NaN)")
pd.DataFrame(dur).to_csv("time_matrix_sec.csv", index=False, header=False)
pd.DataFrame(dist).to_csv("distance_matrix_m.csv", index=False, header=False)
print("сохранено: time_matrix_sec.csv, distance_matrix_m.csv", dur.shape)
