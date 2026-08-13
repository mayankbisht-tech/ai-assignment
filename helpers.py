"""Helper utilities for capacities, splitting parties, and catalog lookups."""
from typing import Tuple, List, Dict


def max_capacities(catalog: List[Dict]) -> Tuple[int, int]:
    """Return (max_hotel_capacity, max_transport_capacity) from catalog."""
    max_hotel = 0
    max_trans = 0
    for item in catalog:
        if item.get("type") == "hotel":
            max_hotel = max(max_hotel, int(item.get("capacity", 0)))
        if item.get("type") == "transport":
            max_trans = max(max_trans, int(item.get("capacity", 0)))
    return max_hotel, max_trans


def rooms_needed(total_people: int, hotel_capacity: int) -> int:
    if hotel_capacity <= 0:
        return 0
    return -(-total_people // hotel_capacity)


def vehicles_needed(total_people: int, transport_capacity: int) -> int:
    if transport_capacity <= 0:
        return 0
    return -(-total_people // transport_capacity)


def find_catalog_item_by_id(catalog: List[Dict], item_id: str) -> Dict | None:
    for item in catalog:
        if item.get("id") == item_id:
            return item
    return None
