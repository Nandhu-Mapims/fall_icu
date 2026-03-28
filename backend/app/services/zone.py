# Purpose: Utilities to evaluate ICU bed-area mapping and patient position against a polygon zone.
from __future__ import annotations

from typing import Iterable

from app.schemas import Point


def parse_polygon(raw_polygon: str) -> list[Point]:
    points: list[Point] = []
    for pair in (raw_polygon or "").split(";"):
        if "," not in pair:
            continue
        x_raw, y_raw = pair.split(",", maxsplit=1)
        try:
            point = Point(x=float(x_raw), y=float(y_raw))
            points.append(point)
        except ValueError:
            continue
    return points


def point_in_polygon(point: Point, polygon: Iterable[Point]) -> bool:
    vertices = list(polygon)
    if len(vertices) < 3:
        return False

    inside = False
    j = len(vertices) - 1
    for i in range(len(vertices)):
        xi, yi = vertices[i].x, vertices[i].y
        xj, yj = vertices[j].x, vertices[j].y
        denominator = (yj - yi) or 1e-9
        intersects = (yi > point.y) != (yj > point.y)
        intersects = intersects and (point.x < ((xj - xi) * (point.y - yi) / denominator + xi))
        if intersects:
            inside = not inside
        j = i
    return inside
