"""
EXPERIMENT 14 - Word Count using MapReduce
Includes Hadoop Streaming mapper/reducer modes and a local simulation
of map, combiner, shuffle-sort, and reduce phases.

Run local simulation:
    python experiment14_mapreduce_word_count.py

Test mapper/reducer with a pipe:
    cat news.txt | python experiment14_mapreduce_word_count.py map | sort |
    python experiment14_mapreduce_word_count.py reduce
"""

import re
import sys
from itertools import groupby


def mapper(lines):
    for line in lines:
        for word in re.findall(r"[a-z]+", line.lower()):
            yield f"{word}\t1"


def reducer(lines):
    pairs = (line.rstrip("\n").split("\t") for line in lines if line.strip())
    for word, group in groupby(pairs, key=lambda item: item[0]):
        yield f"{word}\t{sum(int(count) for _, count in group)}"


def local_simulation():
    splits = {
        "split-1 (node1)": [
            "Data science uses data to create value",
            "Big data needs distributed storage",
        ],
        "split-2 (node2)": [
            "Hadoop stores big data in HDFS",
            "MapReduce processes data in parallel",
        ],
        "split-3 (node3)": [
            "Spark and Hadoop process big data",
            "Data drives science",
        ],
    }

    print("MAPREDUCE WORD COUNT")
    mapped_by_split = {}
    print("\n--- MAP PHASE ---")
    for split, lines in splits.items():
        mapped = list(mapper(lines))
        mapped_by_split[split] = mapped
        print(f"{split}: {len(mapped)} pairs")
        print("  " + ", ".join(x.replace("\t", ":") for x in mapped))

    print("\n--- COMBINER PHASE ---")
    combined = []
    for split, mapped in mapped_by_split.items():
        local = list(reducer(sorted(mapped)))
        combined.extend(local)
        print(f"{split}: {len(mapped)} pairs -> {len(local)} combined pairs")

    print("\n--- SHUFFLE AND SORT PHASE ---")
    shuffled = sorted(combined, key=lambda item: item.split("\t")[0])
    for word, group in groupby(shuffled, key=lambda item: item.split("\t")[0]):
        values = [item.split("\t")[1] for item in group]
        print(f"{word:12} -> {values}")

    print("\n--- REDUCE PHASE ---")
    results = [line.split("\t") for line in reducer(shuffled)]
    results.sort(key=lambda item: (-int(item[1]), item[0]))
    for word, count in results:
        print(f"{word:12} {count}")

    print("\nDistinct words:", len(results))
    print("Total words:", sum(int(count) for _, count in results))
    print("Top 3 words:", [word for word, _ in results[:3]])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "map":
        for result in mapper(sys.stdin):
            print(result)
    elif len(sys.argv) > 1 and sys.argv[1] == "reduce":
        for result in reducer(sys.stdin):
            print(result)
    else:
        local_simulation()
