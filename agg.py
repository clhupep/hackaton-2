import pandas as pd

SRC = "locations.csv"
DST = "locations_agg.csv"   # перезаписываем; если боишься — поставь "locations_agg.csv"

loc = pd.read_csv(SRC, encoding="utf-8-sig")
orig_len = len(loc)

# 1) ключ агрегации — координаты с точностью 6 знаков
loc["_lat_r"] = loc["lat"].round(6)
loc["_lon_r"] = loc["lon"].round(6)

# 2) чтобы депо (id=0) гарантированно остался первым и не слился с другой точкой
#    (на случай, если у депо координаты совпадут с чьими-то ещё — обычно нет, но пусть)
loc["_is_depot"] = (loc["id"] == 0).astype(int)

# 3) агрегируем
def join_names(s):
    return " ".join(str(x).strip() for x in s if pd.notna(x) and str(x).strip())

agg = (
    loc.groupby(["_is_depot", "_lat_r", "_lon_r"], sort=False, as_index=False)
    .agg(
        id=("id", "first"),
        организация=("организация", join_names),
        адрес=("адрес", "first"),
        lat=("lat", "first"),
        lon=("lon", "first"),
        demand_kg=("demand_kg", "sum"),
        fixed_time_min=("fixed_time_min", "first"),   # первое непустое не гарантируется,
                                                       # поэтому ниже чистим NaN отдельно
        only_1500kg=("only_1500kg", "max"),
    )
)

# 4) fixed_time_min: "первое значение, если есть" — берём первое НЕ NaN
fixed_first = (
    loc.dropna(subset=["fixed_time_min"])
       .groupby(["_is_depot", "_lat_r", "_lon_r"], sort=False)["fixed_time_min"]
       .first()
)
agg = agg.merge(
    fixed_first.rename("fixed_first").reset_index(),
    on=["_is_depot", "_lat_r", "_lon_r"],
    how="left",
)
agg["fixed_time_min"] = agg["fixed_first"]
agg = agg.drop(columns=["fixed_first"])

# 5) порядок строк: депо первым, дальше — как в исходном файле
agg = agg.sort_values(["_is_depot", "id"], ascending=[False, True], kind="stable").reset_index(drop=True)

# 6) переназначаем id подряд 0..n-1 (чтобы id == позиция строки == индекс в матрице)
agg["id"] = range(len(agg))

# 7) приводим типы и порядок колонок к исходному
agg["demand_kg"] = agg["demand_kg"].astype(int)
agg["only_1500kg"] = agg["only_1500kg"].astype(int)
agg["fixed_time_min"] = agg["fixed_time_min"].astype("Int64")  # целые с NaN

out = agg[["id", "организация", "адрес", "lat", "lon",
           "demand_kg", "fixed_time_min", "only_1500kg"]]

out.to_csv(DST, index=False, encoding="utf-8-sig")
print(f"было строк: {orig_len}, стало: {len(out)}")
print(out.head(10).to_string(index=False))