import math
import random
import heapq
from dataclasses import dataclass

from vector_core import VectorItem, get_distance_function


@dataclass
class HNSWNode:
    item: VectorItem
    max_level: int
    neighbors: list[list[int]]


class HNSW:
    """
    Hierarchical Navigable Small World graph.

    Reimplementation of the HNSW index used by the
    original NeuroVector C++ project.
    """

    def __init__(
        self,
        m: int = 16,
        ef_build: int = 200,
        metric: str = "euclidean"
    ):
        self.M = m
        self.M0 = 2 * m
        self.ef_build = ef_build

        self.mL = 1.0 / math.log(m)

        self.graph: dict[int, HNSWNode] = {}

        self.top_layer = -1
        self.entry_point = -1

        # Same deterministic seed as the original project.
        self.rng = random.Random(42)

        self.distance_function = get_distance_function(metric)

    # --------------------------------------------------
    # Random level
    # --------------------------------------------------

    def random_level(self) -> int:
        u = self.rng.random()

        return int(
            math.floor(
                -math.log(u) * self.mL
            )
        )

    # --------------------------------------------------
    # Search a single HNSW layer
    # --------------------------------------------------

    def search_layer(
        self,
        query: list[float],
        entry_point: int,
        ef: int,
        layer: int
    ) -> list[tuple[float, int]]:

        if entry_point not in self.graph:
            return []

        visited = set()

        candidates = []
        found = []

        initial_distance = self.distance_function(
            query,
            self.graph[entry_point].item.emb
        )

        heapq.heappush(
            candidates,
            (initial_distance, entry_point)
        )

        # Max-heap implemented using negative distances.
        heapq.heappush(
            found,
            (-initial_distance, entry_point)
        )

        visited.add(entry_point)

        while candidates:

            current_distance, current_id = heapq.heappop(
                candidates
            )

            farthest_distance = -found[0][0]

            if (
                len(found) >= ef
                and current_distance > farthest_distance
            ):
                break

            node = self.graph.get(current_id)

            if node is None:
                continue

            if layer >= len(node.neighbors):
                continue

            for neighbor_id in node.neighbors[layer]:

                if neighbor_id in visited:
                    continue

                if neighbor_id not in self.graph:
                    continue

                visited.add(neighbor_id)

                neighbor = self.graph[neighbor_id]

                neighbor_distance = self.distance_function(
                    query,
                    neighbor.item.emb
                )

                if (
                    len(found) < ef
                    or neighbor_distance < -found[0][0]
                ):
                    heapq.heappush(
                        candidates,
                        (neighbor_distance, neighbor_id)
                    )

                    heapq.heappush(
                        found,
                        (-neighbor_distance, neighbor_id)
                    )

                    if len(found) > ef:
                        heapq.heappop(found)

        results = [
            (-distance, node_id)
            for distance, node_id in found
        ]

        results.sort(
            key=lambda x: (x[0], x[1])
        )

        return results

    # --------------------------------------------------
    # Select nearest neighbors
    # --------------------------------------------------

    def select_neighbors(
        self,
        query: list[float],
        candidates: list[tuple[float, int]],
        max_neighbors: int
    ) -> list[int]:

        candidates = sorted(
            candidates,
            key=lambda x: (x[0], x[1])
        )

        selected = []

        for distance, node_id in candidates:

            if node_id not in self.graph:
                continue

            selected.append(node_id)

            if len(selected) >= max_neighbors:
                break

        return selected

    # --------------------------------------------------
    # Insert
    # --------------------------------------------------

    def insert(self, item: VectorItem) -> None:

        if item.id in self.graph:
            return

        level = self.random_level()

        node = HNSWNode(
            item=item,
            max_level=level,
            neighbors=[
                []
                for _ in range(level + 1)
            ]
        )

        # First node.
        if self.entry_point == -1:

            self.graph[item.id] = node

            self.entry_point = item.id
            self.top_layer = level

            return

        current_entry = self.entry_point
        old_top_layer = self.top_layer

        # Add node before graph searches.
        self.graph[item.id] = node

        # Search existing upper layers first.
        for current_level in range(
            old_top_layer,
            level,
            -1
        ):

            if current_entry not in self.graph:
                break

            entry_node = self.graph[current_entry]

            if current_level >= len(
                entry_node.neighbors
            ):
                continue

            results = self.search_layer(
                item.emb,
                current_entry,
                1,
                current_level
            )

            if results:
                current_entry = results[0][1]

        # Connect the new node from its own highest layer
        # down to layer zero.
        for current_level in range(
            min(level, old_top_layer),
            -1,
            -1
        ):

            results = self.search_layer(
                item.emb,
                current_entry,
                self.ef_build,
                current_level
            )

            max_neighbors = (
                self.M0
                if current_level == 0
                else self.M
            )

            selected = self.select_neighbors(
                item.emb,
                results,
                max_neighbors
            )

            node.neighbors[current_level].extend(
                selected
            )

            # Bidirectional connections.
            for neighbor_id in selected:

                neighbor = self.graph.get(
                    neighbor_id
                )

                if neighbor is None:
                    continue

                if current_level >= len(
                    neighbor.neighbors
                ):
                    continue

                neighbor.neighbors[
                    current_level
                ].append(item.id)

                # Prune oversized neighbor list.
                if len(
                    neighbor.neighbors[current_level]
                ) > max_neighbors:

                    neighbor_candidates = []

                    for neighbor_candidate_id in (
                        neighbor.neighbors[current_level]
                    ):

                        if neighbor_candidate_id not in self.graph:
                            continue

                        candidate_node = self.graph[
                            neighbor_candidate_id
                        ]

                        candidate_distance = (
                            self.distance_function(
                                neighbor.item.emb,
                                candidate_node.item.emb
                            )
                        )

                        neighbor_candidates.append(
                            (
                                candidate_distance,
                                neighbor_candidate_id
                            )
                        )

                    neighbor.neighbors[
                        current_level
                    ] = self.select_neighbors(
                        neighbor.item.emb,
                        neighbor_candidates,
                        max_neighbors
                    )

            if results:
                current_entry = results[0][1]

        # A node with a higher level becomes the new entry point.
        if level > old_top_layer:
            self.top_layer = level
            self.entry_point = item.id

    # --------------------------------------------------
    # Search
    # --------------------------------------------------

    def search(
        self,
        query: list[float],
        k: int = 5,
        ef: int = 50
    ) -> list[tuple[VectorItem, float]]:

        if not self.graph:
            return []

        current_entry = self.entry_point

        # Greedy search through upper layers.
        for layer in range(
            self.top_layer,
            0,
            -1
        ):

            results = self.search_layer(
                query,
                current_entry,
                1,
                layer
            )

            if results:
                current_entry = results[0][1]

        # Full search at layer zero.
        results = self.search_layer(
            query,
            current_entry,
            max(ef, k),
            0
        )

        final_results = []

        for distance, node_id in results[:k]:

            node = self.graph.get(node_id)

            if node is not None:
                final_results.append(
                    (node.item, distance)
                )

        return final_results

    # --------------------------------------------------
    # Remove
    # --------------------------------------------------

    def remove(self, item_id: int) -> bool:

        if item_id not in self.graph:
            return False

        # Remove references from every node.
        for node in self.graph.values():

            for layer in range(
                len(node.neighbors)
            ):

                node.neighbors[layer] = [
                    neighbor_id
                    for neighbor_id in node.neighbors[layer]
                    if neighbor_id != item_id
                ]

        # Remove node itself.
        del self.graph[item_id]

        # Empty graph.
        if not self.graph:
            self.entry_point = -1
            self.top_layer = -1
            return True

        # If the entry point was removed,
        # choose another node.
        if self.entry_point == item_id:

            best_node = max(
                self.graph.values(),
                key=lambda node: node.max_level
            )

            self.entry_point = best_node.item.id
            self.top_layer = best_node.max_level

        return True

    # --------------------------------------------------
    # HNSW graph information
    # --------------------------------------------------

    def get_info(self) -> dict:

        nodes_per_layer = []
        edges_per_layer = []

        for layer in range(
            max(0, self.top_layer + 1)
        ):

            node_count = 0
            edge_count = 0

            for node in self.graph.values():

                if layer < len(node.neighbors):

                    node_count += 1
                    edge_count += len(
                        node.neighbors[layer]
                    )

            nodes_per_layer.append(node_count)
            edges_per_layer.append(edge_count)

        total_edges = sum(
            edges_per_layer
        )

        return {
            "topLayer": self.top_layer,
            "nodeCount": len(self.graph),
            "nodesPerLayer": nodes_per_layer,
            "edgesPerLayer": edges_per_layer,
            "nodes": len(self.graph),
            "edges": total_edges
        }

    # --------------------------------------------------
    # Size
    # --------------------------------------------------

    def size(self) -> int:
        return len(self.graph)