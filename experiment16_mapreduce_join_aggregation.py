"""
EXPERIMENT 16 - Relational Join and Aggregation using MapReduce
Job 1 performs a reduce-side inner join of customers and orders.
Job 2 groups joined orders by city and calculates revenue, order count,
maximum order value, and average order value.

Run: python experiment16_mapreduce_join_aggregation.py
"""


CUSTOMERS = [
    "C1,Kavya,Chennai",
    "C2,Ravi,Bengaluru",
    "C3,Sana,Chennai",
    "C4,Tarun,Hyderabad",
    "C5,Uma,Bengaluru",
]

ORDERS = [
    "O1,C1,2500",
    "O2,C2,1800",
    "O3,C1,700",
    "O4,C3,4200",
    "O5,C4,1500",
    "O6,C2,3100",
    "O7,C5,950",
    "O8,C3,600",
    "O9,C9,800",  # No matching customer; dropped by the inner join.
]


def join_mapper(source, line):
    fields = line.split(",")
    if source == "customers":
        return fields[0], ("C", fields[1], fields[2])
    return fields[1], ("O", fields[0], int(fields[2]))


def group_by_key(pairs):
    groups = {}
    for key, value in pairs:
        groups.setdefault(key, []).append(value)
    return sorted(groups.items())


def join_reducer(customer_id, tagged_values):
    customers = [value for value in tagged_values if value[0] == "C"]
    orders = [value for value in tagged_values if value[0] == "O"]
    # Inner join: output rows only if both customer and order exist.
    for customer in customers:
        for order in orders:
            yield (customer_id, customer[1], customer[2], order[1], order[2])


def aggregation_mapper(joined_row):
    _, _, city, _, amount = joined_row
    return city, amount


def aggregation_reducer(city, amounts):
    return {
        "city": city,
        "revenue": sum(amounts),
        "orders": len(amounts),
        "maximum": max(amounts),
        "average": sum(amounts) / len(amounts),
    }


def main():
    print("RELATIONAL JOIN AND AGGREGATION USING MAPREDUCE")

    print("\n--- JOB 1: MAP PHASE ---")
    mapped = (
        [join_mapper("customers", row) for row in CUSTOMERS]
        + [join_mapper("orders", row) for row in ORDERS]
    )
    for key, value in mapped:
        print(f"{key} -> {value}")

    print("\n--- JOB 1: SHUFFLE AND SORT PHASE ---")
    groups = group_by_key(mapped)
    for key, values in groups:
        print(f"{key}: {values}")

    print("\n--- JOB 1: REDUCE-SIDE INNER JOIN ---")
    joined = [
        row
        for customer_id, values in groups
        for row in join_reducer(customer_id, values)
    ]
    print(f"{'Customer':10} {'Name':8} {'City':12} {'Order':8} Amount")
    for row in joined:
        print(f"{row[0]:10} {row[1]:8} {row[2]:12} {row[3]:8} {row[4]}")
    print("Order O9 is dropped because customer C9 does not exist.")

    print("\n--- JOB 2: MAP PHASE ---")
    mapped2 = [aggregation_mapper(row) for row in joined]
    print(mapped2)

    print("\n--- JOB 2: SHUFFLE AND SORT PHASE ---")
    city_groups = group_by_key(mapped2)
    for city, amounts in city_groups:
        print(f"{city}: {amounts}")

    print("\n--- JOB 2: REDUCE / CITY-WISE AGGREGATION ---")
    reports = [aggregation_reducer(city, amounts)
               for city, amounts in city_groups]
    reports.sort(key=lambda report: report["revenue"], reverse=True)
    print(f"{'City':12} {'Revenue':>10} {'Orders':>8} {'Maximum':>9} {'Average':>10}")
    for report in reports:
        print(f"{report['city']:12} {report['revenue']:10d} "
              f"{report['orders']:8d} {report['maximum']:9d} "
              f"{report['average']:10.2f}")

    print("\nGrand total revenue:",
          sum(report["revenue"] for report in reports))


if __name__ == "__main__":
    main()
