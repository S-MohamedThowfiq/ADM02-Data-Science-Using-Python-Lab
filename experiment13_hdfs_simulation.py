"""
EXPERIMENT 13 - Distributed File System and Loading Data into HDFS
Simulates a six-node, two-rack HDFS cluster, block replication, reading,
checksum verification, and recovery after a DataNode failure.
Run: python experiment13_hdfs_simulation.py
"""

import hashlib

BLOCK_SIZE = 64
REPLICATION = 3


class DataNode:
    def __init__(self, name, rack):
        self.name = name
        self.rack = rack
        self.alive = True
        self.blocks = {}


class NameNode:
    """Stores metadata: file path -> block IDs and block ID -> DataNodes."""

    def __init__(self, datanodes):
        self.datanodes = datanodes
        self.namespace = {}
        self.block_map = {}
        self.next_block_id = 1

    def node(self, name):
        return next(d for d in self.datanodes if d.name == name)

    def choose_targets(self):
        live = [d for d in self.datanodes if d.alive]
        if len(live) < REPLICATION:
            raise RuntimeError("Not enough live DataNodes for three replicas.")

        # Rack-aware placement: one replica on the first rack and two on
        # the other rack. Rotate the first node to distribute the workload.
        index = (self.next_block_id - 1) % len(live)
        first = live[index]
        other_rack = [d for d in live if d.rack != first.rack]
        same_other_rack = other_rack[:]
        if len(same_other_rack) < 2:
            raise RuntimeError("Rack-aware placement needs two nodes on the other rack.")
        return [first, same_other_rack[0], same_other_rack[1]]

    def put(self, path, data):
        block_ids = []
        for start in range(0, len(data), BLOCK_SIZE):
            chunk = data[start:start + BLOCK_SIZE]
            block_id = f"blk_{self.next_block_id:04d}"
            targets = self.choose_targets()
            for dn in targets:
                dn.blocks[block_id] = chunk
            self.block_map[block_id] = [dn.name for dn in targets]
            block_ids.append(block_id)
            self.next_block_id += 1
        self.namespace[path] = block_ids
        return block_ids

    def get(self, path):
        chunks = []
        for block_id in self.namespace[path]:
            available = None
            for name in self.block_map[block_id]:
                dn = self.node(name)
                if dn.alive and block_id in dn.blocks:
                    available = dn.blocks[block_id]
                    break
            if available is None:
                raise IOError(f"No live replica available for {block_id}")
            chunks.append(available)
        return b"".join(chunks)

    def handle_failure(self, name):
        failed = self.node(name)
        failed.alive = False
        repaired = 0

        for block_id, holders in self.block_map.items():
            if name not in holders:
                continue
            holders.remove(name)
            if not holders:
                raise IOError(f"All replicas lost for {block_id}")

            source = next(self.node(n) for n in holders
                          if self.node(n).alive and block_id in self.node(n).blocks)
            live_holders = [n for n in holders if self.node(n).alive]
            candidates = [d for d in self.datanodes
                          if d.alive and d.name not in live_holders]
            if len(live_holders) < REPLICATION and candidates:
                target = candidates[0]
                target.blocks[block_id] = source.blocks[block_id]
                holders.append(target.name)
                repaired += 1
        return repaired


def checksum(data):
    return hashlib.md5(data).hexdigest()


def main():
    cluster = [
        DataNode("DN1", "Rack-1"), DataNode("DN2", "Rack-1"),
        DataNode("DN3", "Rack-1"), DataNode("DN4", "Rack-2"),
        DataNode("DN5", "Rack-2"), DataNode("DN6", "Rack-2"),
    ]
    namenode = NameNode(cluster)

    log_text = "\n".join(
        f"2026-03-0{day} 10:{minute:02d} sensor={sensor} "
        f"temp={30 + (day * minute + sensor) % 9}"
        for day in range(1, 3)
        for minute in (5, 20)
        for sensor in (1, 2)
    )
    log_data = log_text.encode("utf-8")
    path = "/data/sensor_log.txt"

    print("DISTRIBUTED FILE SYSTEM (HDFS) SIMULATION")
    print("Cluster:", ", ".join(f"{d.name}({d.rack})" for d in cluster))
    block_ids = namenode.put(path, log_data)
    print(f"\nPUT {path}: {len(log_data)} bytes, {len(block_ids)} blocks")

    print("\nNameNode block metadata:")
    for bid in block_ids:
        print(f"{bid}: {namenode.block_map[bid]}")

    print("\nDataNode storage report:")
    for dn in cluster:
        print(f"{dn.name}: {sorted(dn.blocks)}")

    downloaded = namenode.get(path)
    print("\nFirst three lines read from HDFS:")
    print("\n".join(downloaded.decode().splitlines()[:3]))
    print("Original checksum:", checksum(log_data))
    print("Read-back checksum:", checksum(downloaded))
    print("Checksum verified:", checksum(log_data) == checksum(downloaded))

    print("\n*** Simulating failure of DN1 ***")
    repaired = namenode.handle_failure("DN1")
    print("Blocks re-replicated:", repaired)
    for bid in block_ids:
        print(f"{bid}: {namenode.block_map[bid]}")

    after_failure = namenode.get(path)
    healthy = all(
        sum(1 for name in holders if namenode.node(name).alive) == REPLICATION
        for holders in namenode.block_map.values()
    )
    print("File readable after failure:", after_failure == log_data)
    print("Checksum after failure:", checksum(after_failure))
    print("HDFS status:", "HEALTHY" if healthy else "UNDER-REPLICATED")


if __name__ == "__main__":
    main()
