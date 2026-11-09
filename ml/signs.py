from ml.config import CLASS_NAMES

SPEED_LIMITS = {0: 20, 1: 30, 2: 50, 3: 60, 4: 70, 5: 80, 7: 100, 8: 120}
END_OF_RESTRICTION = {6, 32, 41, 42}
NO_ENTRY = {15, 17}
PROHIBITORY = {9, 10, 16}
PRIORITY = {11, 12}
WARNING = set(range(18, 32))
MANDATORY = set(range(33, 41))
YIELD, STOP = 13, 14


def _info(category: str, action: str, speed_limit_kmh: int | None = None) -> dict:
    return {"category": category, "action": action, "speed_limit_kmh": speed_limit_kmh}


def describe(class_id: int) -> dict:
    name = CLASS_NAMES[class_id]
    if class_id in SPEED_LIMITS:
        limit = SPEED_LIMITS[class_id]
        return _info("speed_limit", f"Set target speed to {limit} km/h", limit)
    if class_id == STOP:
        return _info("stop", "Come to a full stop at the line, proceed when clear")
    if class_id == YIELD:
        return _info("yield", "Slow down and give way to crossing traffic")
    if class_id in PRIORITY:
        return _info("priority", "Maintain speed, you have right of way")
    if class_id in NO_ENTRY:
        return _info("no_entry", "Do not enter, plan an alternative route")
    if class_id in PROHIBITORY:
        return _info("prohibitory", f"Restriction active: {name}")
    if class_id in END_OF_RESTRICTION:
        return _info("end_of_restriction", f"{name}: restore default behaviour")
    if class_id in WARNING:
        return _info("warning", f"Hazard ahead ({name}), reduce speed")
    if class_id in MANDATORY:
        return _info("mandatory", f"Follow mandatory direction: {name}")
    return _info("other", name)
