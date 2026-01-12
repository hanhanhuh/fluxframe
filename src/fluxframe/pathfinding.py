#!/usr/bin/env python3
"""Pathfinding module for generating videos via perceptual similarity."""

from __future__ import annotations

import contextlib
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .config import Config
from .database import ImageDatabase
from .search import SearchIndex


class PathFinder:
    """Pathfinding engine using leashed backtracking with smoothing."""

    # Maximum frames to backtrack before forcing a move
    MAX_BACKTRACK_DEPTH = 50

    def __init__(self, cfg: Config, db: ImageDatabase, idx: SearchIndex):
        """Initialize path finder.

        Args:
            cfg: Configuration object
            db: Image database
            idx: Search index
        """
        self.cfg = cfg
        self.db = db
        self.idx = idx
        self.k_candidates = 200  # Starting k value, adjusts dynamically
        self.consecutive_failures = 0  # Track failures to adjust k

    def find_path(self) -> list[int]:
        """Find path through image space using leashed backtracking.

        This algorithm uses:
        - Interactive start selection (visual search fallback)
        - Auto-threshold adjustment
        - Greedy search with limited backtracking
        - Smoothing via momentum (average of last K frames)

        Returns:
            List of image indices forming the path
        """
        # 1. Select start node
        start_node = self._select_start_node()

        # 2. Auto-adjust threshold if needed
        self._auto_adjust_threshold(start_node)

        # 3. Run main pathfinding loop
        return self._search_loop(start_node)

    def _select_start_node(self) -> int:
        """Select starting image interactively or randomly.

        Returns:
            Index of starting image
        """
        start_node = -1

        if self.cfg.start_filename:
            target_name = Path(self.cfg.start_filename).name

            # Check 1: Exact filename match in cache
            if target_name in self.db.filenames:
                start_node = self.db.filenames.index(target_name)
                print(f"[Algo] Start image found in cache: {target_name}")
            else:
                # Check 2: Visual search fallback (interactive)
                start_node = self._visual_search_fallback(target_name)

        # Fallback to random if no start specified
        if start_node == -1:
            start_node = random.randint(0, len(self.db.filenames) - 1)
            print(f"[Algo] Random start: {self.db.filenames[start_node]}")

        return start_node

    def _visual_search_fallback(self, target_name: str) -> int:
        """Perform visual search when exact filename not found.

        Displays top 10 candidates and prompts user to select one.
        Calls sys.exit() if image cannot be loaded.

        Args:
            target_name: Name of target file

        Returns:
            Index of selected image from user choice.

        Raises:
            SystemExit: If start image file cannot be loaded.
        """
        print(f"[Algo] '{target_name}' not found in cache.")
        print("[Algo] Analyzing image content and creating preview...")

        # Load and process query image
        img = cv2.imread(self.cfg.start_filename)  # type: ignore
        if img is None:
            sys.exit(f"Error: Could not load {self.cfg.start_filename}")

        small = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32).flatten()
        query_vec = (lab * self.idx.w_tile).reshape(1, -1)

        # Search for top candidates
        _distances, _indices = self.idx.index.search(query_vec, 10)  # type: ignore

        # Create preview directory
        preview_dir = self.cfg.output_dir / "_CANDIDATE_PREVIEW"
        if preview_dir.exists():
            shutil.rmtree(preview_dir)
        preview_dir.mkdir(exist_ok=True)

        # Display candidates
        print(f"\n{'=' * 80}")
        print(" NOTICE: Start image not found exactly.")
        print(f" Top 10 candidates saved to: {preview_dir}")
        print(f"{'=' * 80}")

        candidates = []
        for rank, (dist, node_idx) in enumerate(zip(_distances[0], _indices[0], strict=False)):
            if node_idx == -1:
                continue

            fname = self.db.filenames[node_idx]
            full_path = self.cfg.img_dir / fname
            candidates.append(node_idx)

            # Copy to preview directory
            dest_name = f"{rank}_{fname}"
            with contextlib.suppress(Exception):
                shutil.copy(full_path, preview_dir / dest_name)

            print(f" [{rank}] {full_path.absolute()} (Dist: {dist:.0f})")

        print(f"{'-' * 80}")
        choice = input("Select number (0-9) [Enter=0]: ").strip()

        # Parse user choice
        selected_rank = 0
        if choice:
            try:
                selected_rank = int(choice)
                if selected_rank < 0 or selected_rank >= len(candidates):
                    selected_rank = 0
            except ValueError:
                selected_rank = 0

        start_node = candidates[selected_rank]
        print(f"[Algo] Start image manually set: {self.db.filenames[start_node]}")

        return start_node

    def _auto_adjust_threshold(self, start_node: int) -> None:
        """Automatically adjust threshold based on dataset density.

        Modifies cfg.threshold if current value is too restrictive.

        Args:
            start_node: Starting node index
        """
        # Check distances to neighbors
        dists_check, _ = self.idx.search_id(start_node, k_candidates=50)

        if len(dists_check) > 10:
            ref_dist = dists_check[10]

            if self.cfg.threshold < ref_dist:
                suggested = ref_dist * 2
                print(f"[Algo] Auto-tune: Threshold {self.cfg.threshold:.0f} -> {suggested:.0f}")
                self.cfg.threshold = suggested

    def _adjust_k_candidates(self, current_frame: int) -> None:
        """Dynamically adjust k_candidates based on search difficulty.

        Increases k when struggling to find candidates, decreases when finding easily.

        Args:
            current_frame: Current frame number in path
        """
        # Increase k if we're having trouble finding candidates
        if self.consecutive_failures >= 3:
            old_k = self.k_candidates
            self.k_candidates = min(self.k_candidates * 2, len(self.db.filenames))
            if old_k != self.k_candidates:
                print(f"[Algo] Expanding search: k={old_k} -> {self.k_candidates}")
            self.consecutive_failures = 0

        # Decrease k periodically if we've been succeeding (every 1000 frames)
        # This keeps searches efficient when we move to a new dense region
        elif current_frame % 1000 == 0 and self.k_candidates > 200:
            old_k = self.k_candidates
            self.k_candidates = max(200, self.k_candidates // 2)
            if old_k != self.k_candidates:
                print(f"[Algo] Reducing search: k={old_k} -> {self.k_candidates}")

    def _search_loop(self, start_node: int) -> list[int]:
        """Main pathfinding loop with leashed backtracking.

        Args:
            start_node: Starting node index

        Returns:
            List of image indices forming the generated path.
        """
        print(f"[Algo] Starting pathfinding from: {self.db.filenames[start_node]}")
        print(f"[Algo] Smoothing (Momentum): Last {self.cfg.smoothing_k} frames")
        print(f"[Algo] Dynamic k_candidates: starts at {self.k_candidates}")

        # Initialize state
        path = [start_node]
        visited = {start_node}

        # Initialize stack with ID search
        d, i = self.idx.search_id(start_node, k_candidates=self.k_candidates)
        stack = [[start_node, i, d, 0]]

        max_reached_frame = 1

        pbar = tqdm(total=self.cfg.total_frames, desc="Pathfinding")
        pbar.update(1)

        # Main loop
        while len(path) < self.cfg.total_frames:
            max_reached_frame = max(max_reached_frame, len(path))

            # Dynamically adjust k_candidates based on progress
            self._adjust_k_candidates(len(path))

            # Emergency stack refill
            if not stack:
                if not self._emergency_stack_refill(path, visited, stack, pbar):
                    break
                continue

            # Try to extend path
            if not self._try_extend_path(path, visited, stack, pbar, max_reached_frame):
                break

        pbar.close()
        return path

    def _emergency_stack_refill(
        self, path: list[int], visited: set[int], stack: list, pbar: tqdm
    ) -> bool:
        """Refill stack in emergency when completely empty.

        Args:
            path: Current path
            visited: Set of visited nodes
            stack: Search stack
            pbar: Progress bar

        Returns:
            True if found unvisited neighbor, False if all images exhausted.
        """
        last = path[-1]

        # Start with current k_candidates, then expand if needed
        # Always include database size as final fallback
        k_sizes = [self.k_candidates]
        for k in [1000, 5000, 10000, 50000, 100000, len(self.db.filenames)]:
            if k > self.k_candidates:
                k_sizes.append(k)

        # Ensure we always try at least the full database
        if k_sizes[-1] < len(self.db.filenames):
            k_sizes.append(len(self.db.filenames))

        for k_size in k_sizes:
            _d, i = self.idx.search_id(last, k_candidates=min(k_size, len(self.db.filenames)))

            for cand in i:
                if cand not in visited:
                    visited.add(cand)
                    path.append(cand)

                    # Reset stack with ID search
                    dn, in_ = self.idx.search_id(cand, k_candidates=self.k_candidates)
                    stack.append([cand, in_, dn, 0])
                    pbar.update(1)
                    self.consecutive_failures = 0  # Success - reset counter
                    return True

        # All images have been visited
        self.consecutive_failures += 1
        return False

    def _try_extend_path(
        self,
        path: list[int],
        visited: set[int],
        stack: list,
        pbar: tqdm,
        max_reached_frame: int,
    ) -> bool:
        """Try to extend path by one frame.

        Args:
            path: Current path
            visited: Set of visited nodes
            stack: Search stack
            pbar: Progress bar
            max_reached_frame: Maximum frame count reached so far

        Returns:
            True to continue searching, False to terminate.
        """
        current_state = stack[-1]
        c_inds, c_dists = current_state[1], current_state[2]
        ptr = current_state[3]

        found_next = False

        # Try candidates within threshold
        for k in range(ptr, len(c_inds)):
            cand = c_inds[k]
            dist = c_dists[k]
            current_state[3] = k + 1

            if cand != -1 and cand not in visited and dist < self.cfg.threshold:
                # Found valid next frame
                visited.add(cand)
                path.append(cand)

                # Apply smoothing: search from average of last K frames
                window_indices = path[-self.cfg.smoothing_k :]
                vectors = self.db.data[window_indices].astype(np.float32)  # type: ignore
                mean_vec = np.mean(vectors, axis=0)

                new_d, new_i = self.idx.search_vector(mean_vec, k_candidates=self.k_candidates)
                stack.append([cand, new_i, new_d, 0])

                found_next = True
                self.consecutive_failures = 0  # Success - reset counter
                pbar.update(1)
                break

        if not found_next:
            # Backtracking or forced move required
            return self._handle_dead_end(path, visited, stack, pbar, max_reached_frame)

        return True

    def _handle_dead_end(
        self,
        path: list[int],
        visited: set[int],
        stack: list,
        pbar: tqdm,
        max_reached_frame: int,
    ) -> bool:
        """Handle dead end via backtracking or forced move.

        Args:
            path: Current path
            visited: Set of visited nodes
            stack: Search stack
            pbar: Progress bar
            max_reached_frame: Maximum frame count reached so far

        Returns:
            True to continue searching, False to terminate.
        """
        frames_lost = max_reached_frame - len(path)

        # Within leash length: backtrack
        if frames_lost < self.MAX_BACKTRACK_DEPTH:
            if len(path) > 1:
                path.pop()
                stack.pop()
                pbar.update(-1)
                return True
            # Can't backtrack from start - force a move instead
            return self._force_move(path, visited, stack, pbar)
        # Beyond leash: force move to escape local minimum
        return self._force_move(path, visited, stack, pbar)

    def _force_move(self, path: list[int], visited: set[int], stack: list, pbar: tqdm) -> bool:
        """Force move to escape local minimum (beyond leash).

        Args:
            path: Current path
            visited: Set of visited nodes
            stack: Search stack
            pbar: Progress bar

        Returns:
            True if found unvisited candidate, False if all images exhausted.
        """
        current_state = stack[-1]
        c_inds = current_state[1]

        # First try current candidates (ignore threshold but require unvisited)
        for k in range(len(c_inds)):
            cand = c_inds[k]
            if cand != -1 and cand not in visited:
                visited.add(cand)
                path.append(cand)

                # Apply smoothing
                window_indices = path[-self.cfg.smoothing_k :]
                vectors = self.db.data[window_indices].astype(np.float32)  # type: ignore
                mean_vec = np.mean(vectors, axis=0)
                dn, in_ = self.idx.search_vector(mean_vec, k_candidates=self.k_candidates)

                stack.append([cand, in_, dn, 0])
                self.consecutive_failures = 0  # Success - reset counter
                pbar.update(1)
                return True

        # Current candidates all visited - search progressively larger neighborhoods
        window_indices = path[-self.cfg.smoothing_k :]
        vectors = self.db.data[window_indices].astype(np.float32)  # type: ignore
        mean_vec = np.mean(vectors, axis=0)

        # Build list of k sizes to try, starting from current k_candidates
        # Always include database size as final fallback
        k_sizes = []
        for k in [1000, 5000, 10000, 50000, 100000, len(self.db.filenames)]:
            if k > self.k_candidates:
                k_sizes.append(k)

        # Ensure we always try at least the full database
        if not k_sizes or k_sizes[-1] < len(self.db.filenames):
            k_sizes.append(len(self.db.filenames))

        for k_size in k_sizes:
            new_d, new_i = self.idx.search_vector(mean_vec, k_candidates=min(k_size, len(self.db.filenames)))

            for cand in new_i:
                if cand != -1 and cand not in visited:
                    visited.add(cand)
                    path.append(cand)

                    # Get new neighborhood for this candidate
                    dn, in_ = self.idx.search_vector(mean_vec, k_candidates=self.k_candidates)
                    stack.append([cand, in_, dn, 0])
                    self.consecutive_failures = 0  # Success - reset counter
                    pbar.update(1)
                    return True

        # All images have been visited
        self.consecutive_failures += 1
        return False


def find_path(cfg: Config, db: ImageDatabase, idx: SearchIndex) -> list[int]:
    """Convenience function for pathfinding.

    Args:
        cfg: Configuration object
        db: Image database
        idx: Search index

    Returns:
        List of image indices forming the path
    """
    finder = PathFinder(cfg, db, idx)
    return finder.find_path()
