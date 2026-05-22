

######################## pasted from Assignment 1 - com490.py

import numpy as np
from collections import defaultdict
from scipy.stats import norm

from delay_model import (
    build_delay_distribution_tables,
    collect_distribution_tables,
    DelayLookup,
    DOW_BUCKET_WEEKDAY,
    DOW_BUCKET_SATURDAY,
    DOW_BUCKET_SUNDAY,
)


DAY_COLUMNS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
WALK_SPEED_M_PER_SEC = 50 / 60  # assignment spec: 50m per minute

# Maps day name to the dow_bucket string expected by DelayLookup
_DAY_TO_DOW_BUCKET = {
    'monday':    DOW_BUCKET_WEEKDAY,
    'tuesday':   DOW_BUCKET_WEEKDAY,
    'wednesday': DOW_BUCKET_WEEKDAY,
    'thursday':  DOW_BUCKET_WEEKDAY,
    'friday':    DOW_BUCKET_WEEKDAY,
    'saturday':  DOW_BUCKET_SATURDAY,
    'sunday':    DOW_BUCKET_SUNDAY,
}

# Column indices in the numpy connection array — keep these in sync with
# the column order in prepare_from_data() so we never have magic numbers in the scan loop
C_DEP_STOP = 0
C_DEP_TIME = 1
C_ARR_STOP = 2
C_ARR_TIME = 3
C_TRIP_IDX = 4   # index into self.trip_ids list (numpy can't store strings)
# columns 5..11 are the day-of-week boolean flags (monday=5 .. sunday=11)
C_DAY_BASE  = 5


class JourneyPlanner:

    def __init__(self):
        # populated by prepare_from_data()
        self.stops          = {}   # stop_id (int) -> {name, lat, lon}
        self.walking        = {}   # stop_id -> [(neighbor_id, distance_m)]
        self.trip_ids       = []   # list of trip_id strings, indexed by C_TRIP_IDX
        self.trip_products  = {}   # trip_id (str) -> product_id_clean (str|None)
        self.trip_lines     = {}   # trip_id (str) -> line_text (str|None)
        self.trip_labels = {}   # trip_id -> route_label

        # STEP 3: primary delay model — empirical CDFs via DelayLookup.
        # Falls back to Normal model (delay_features) when None.
        self.delay_lookup   = None  # DelayLookup | None

        # Kept as fallback for when delay_training is unavailable.
        self.delay_features = {}   # (stop_id, hour_of_day) -> (avg_delay_sec, std_delay_sec)

        # core CSA data structures
        self.conn_arr        = None   # np.ndarray (N,12) sorted by dep_time for forward CSA
        self.dep_times       = None   # 1-D view of C_DEP_TIME
        self.arr_sorted      = None   # np.ndarray (N,12) sorted by arr_time desc for backward CSA
        self.arr_sorted_list = None   # list-of-lists cache — avoids repeated .tolist() in hot loop

    def prepare_from_data(self, data):
        import pandas as _pd
        """Load routing data from PreparedData."""
        # stops
        for row in data.stops.collect():
            self.stops[int(row['stop_id'])] = {
                'name': row['stop_name'],
                'lat':  float(row['stop_lat']),
                'lon':  float(row['stop_lon']),
            }
        print(f"Loaded {len(self.stops)} stops")

        # connections — convert via Pandas for efficiency
        conn_pd = data.connections.orderBy('dep_time_sec').toPandas()

        unique_trips = conn_pd['trip_id'].unique()
        self.trip_ids = list(unique_trips)
        trip_id_to_idx = {tid: i for i, tid in enumerate(unique_trips)}
        conn_pd['trip_idx'] = conn_pd['trip_id'].map(trip_id_to_idx)

        if 'line_text' in conn_pd.columns:
            _lt = conn_pd.drop_duplicates(subset='trip_id').set_index('trip_id')['line_text']
            self.trip_lines = {tid: (None if _pd.isna(v) else str(v)) for tid, v in _lt.items()}
        
        if 'route_label' in conn_pd.columns:
            _rl = conn_pd.drop_duplicates(subset='trip_id').set_index('trip_id')['route_label']
            self.trip_labels = {tid: (None if _pd.isna(v) else str(v)) for tid, v in _rl.items()}

        day_cols = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        self.conn_arr = conn_pd[
            ['dep_stop_id', 'dep_time_sec', 'arr_stop_id', 'arr_time_sec', 'trip_idx'] + day_cols
        ].astype('int64').values
        self.dep_times  = self.conn_arr[:, C_DEP_TIME]
        self.arr_sorted      = self.conn_arr[np.argsort(self.conn_arr[:, C_ARR_TIME])[::-1]]
        self.arr_sorted_list = self.arr_sorted.tolist()  # cache once — reused in every CSA scan
        print(f"Loaded {len(self.conn_arr)} connections")

        # walking edges
        for row in data.walking_edges.collect():
            self.walking.setdefault(int(row['a_stop_id']), []).append(
                (int(row['b_stop_id']), float(row['distance_m']))
            )
        print(f"Loaded walking edges for {len(self.walking)} stops")

        # STEP 3: build DelayLookup from delay_training if available.
        if data.delay_training is not None:
            try:
                levels = build_delay_distribution_tables(data.delay_training)
                tables = collect_distribution_tables(levels, min_observations=30)
                self.delay_lookup = DelayLookup.from_collected(tables)
                print("Loaded DelayLookup with empirical distributions")
            except Exception as e:
                print(f"Warning: could not build DelayLookup ({e}). Falling back to Normal model.")
                self.delay_lookup = None

        # Fallback Normal model — loaded regardless so it's available if DelayLookup
        # fails or delay_training is missing.
        if data.delay_features is not None:
            for row in data.delay_features.collect():
                key = (int(row['bpuic']), int(row['hour_of_day']))
                self.delay_features[key] = (
                    float(row['hist_avg_delay_mins']) * 60,  # convert to seconds
                    float(row['hist_std_delay_mins']) * 60,
                )
            print(f"Loaded fallback delay features for {len(self.delay_features)} (stop, hour) pairs")
        else:
            if self.delay_lookup is None:
                print("No delay data available — confidence will default to 1.0")

    def describe_route(self, route_result):
        """Print a single routes() result in human-readable form."""
        dep     = route_result['dep_time']
        arr     = route_result['arr_time']
        conf    = route_result['confidence']
        n_tr    = route_result['n_transfers']
        walk_m  = route_result['walk_m']
        fmt     = lambda s: f"{s // 3600:02d}:{(s % 3600) // 60:02d}"
        print(
            f"Departure {fmt(dep)} → Arrival {fmt(arr)}  "
            f"(confidence {conf:.0%}, {n_tr} transfer(s), {walk_m:.0f}m walking)"
        )
        for ts, s1, trip_id, s2, route_label in route_result['path']:
            t = fmt(ts)
            if trip_id and s2 is None:
                print(f"  {t}  board  {route_label}  at {self.stops.get(s1, {}).get('name', s1)}")
            elif trip_id and s1 is None:
                print(f"  {t}  alight at {self.stops.get(s2, {}).get('name', s2)}")
            else:
                n1 = self.stops.get(s1, {}).get('name', s1)
                n2 = self.stops.get(s2, {}).get('name', s2)
                print(f"  {t}  walk   {n1} → {n2}")

    # ------------------------------------------------------------------
    # STEP 1 — Multi-route Pareto collection
    # ------------------------------------------------------------------
    @staticmethod
    def _trip_sequence(path):
        """Return the ordered tuple of unique consecutive line labels used in a path."""
        seen = []
        for _, _, trip_id, _, route_label in path:
            if trip_id is not None:
                key = route_label or trip_id  # use line label (e.g. "metro m1") not trip instance
                if not seen or seen[-1] != key:
                    seen.append(key)
        return tuple(seen)

    # helper to get human-readable route name
    def _route_label(self, trip_id):
        if trip_id is None:
            return None
        return self.trip_labels.get(trip_id) or self.trip_lines.get(trip_id) or str(trip_id)
    
    def routes(self, start_id, end_id, arrives_by, day,
               confidence_threshold=0.9, max_routes=5,
               min_transfer_secs=120, max_walk_m=500):
        """
        Find up to max_routes routes from start_id to end_id arriving before arrives_by,
        each with confidence >= confidence_threshold.

        Returns a list of dicts sorted from latest departure to earliest:
            [{
                'path':         [...],
                'confidence':   float,
                'dep_time':     int,
                'arr_time':     int,
                'n_transfers':  int,   # STEP 4
                'walk_m':       float, # STEP 4
            }, ...]

        STEP 1: low-confidence routes are skipped but do not terminate the search.
        STEP 2: _backward_csa prunes connections during the scan.
        STEP 3: dow_bucket threaded into empirical delay model.
        STEP 4: n_transfers and walk_m exposed in each result dict.
        """
        if self.arr_sorted is None:
            raise RuntimeError("Call prepare_from_data() first")
        if start_id not in self.stops:
            raise ValueError(f"Unknown start stop_id: {start_id}")
        if end_id not in self.stops:
            raise ValueError(f"Unknown end stop_id: {end_id}")
        if day.lower() not in DAY_COLUMNS:
            raise ValueError(f"day must be one of {DAY_COLUMNS}")
        if not 0 <= max_walk_m <= 500:
            raise ValueError("max_walk_m must be between 0 and 500")

        h, m        = arrives_by.split(':')
        deadline    = int(h) * 3600 + int(m) * 60
        day_col     = C_DAY_BASE + DAY_COLUMNS.index(day.lower())
        dow_bucket  = _DAY_TO_DOW_BUCKET[day.lower()]

        results          = []
        current_max_dep = None
        seen_sequences = set()
        max_attempts     = max_routes * 10
        attempts         = 0

        while len(results) < max_routes and attempts < max_attempts:
            attempts += 1

            path, scan_confidence, n_transfers, walk_m = self._backward_csa(
                start_id, end_id, deadline, day_col,
                min_transfer_secs, max_walk_m,
                confidence_threshold=confidence_threshold,
                dow_bucket=dow_bucket,
                max_dep_time=current_max_dep,
            )
            if not path:
                break

            dep_time = path[0][0]
            arr_time = path[-1][0]
            for event in reversed(path):
                if event[2] is not None and event[3] is not None:  # alight event
                    arr_time = event[0]
                    break
            # Pareto pivot: use the last *transit* event time (alight or board), not
            # arr_time.  When the final segment is a walk, arr_time == current_deadline
            # (T[destination] = deadline), so arr_time - 1 would only decrement by one
            # second each iteration, producing near-duplicate routes.  Walk events have
            # event[2] = None (trip_id); transit events have event[2] != None.
            current_max_dep = dep_time
            trip_seq = self._trip_sequence(path)
            if trip_seq in seen_sequences:
                continue
            seen_sequences.add(trip_seq)
            

            if scan_confidence >= confidence_threshold:

                # when confidence_threshold=0.0, scan_confidence will always be 1.0 
                # because _p_transfer returns 1.0 in deterministic mode, so every label propagates as C=1.0. 
                # This means Q=0 routes will show confidence=100% in the output
                results.append({
                    'path':        path,
                    'confidence':  scan_confidence,
                    'dep_time':    dep_time,
                    'arr_time':    arr_time,
                    'n_transfers': n_transfers,  # STEP 4
                    'walk_m':      walk_m,        # STEP 4
                })

        return results

    # ------------------------------------------------------------------
    # STEP 2 — Confidence integrated into the backward CSA scan
    # STEP 4 — 4-criteria label tracking with dominance pruning
    # ------------------------------------------------------------------
    def _backward_csa(self, start_id, end_id, deadline, day_col,
                      min_transfer_secs, max_walk_m,
                      confidence_threshold=0.0,
                      dow_bucket=DOW_BUCKET_WEEKDAY, max_dep_time=None):
        """
        Backward CSA: find the latest-departing path from start_id to end_id by deadline,
        with confidence >= confidence_threshold at every transfer.

        Each stop label tracks four criteria:
            T[stop]  — latest feasible departure time
            C[stop]  — confidence of reaching end_id on time
            N[stop]  — number of vehicle transfers so far on the path to end_id
            W[stop]  — total walking distance (metres) so far on the path to end_id

        STEP 4 — dominance pruning:
            A new label (dep_time_new, c_new, n_new, w_new) replaces the existing
            label at dep_stop only if it is not dominated. A new label dominates
            the old one when dep_time_new > T[dep_stop] (strictly later departure),
            OR when dep_time_new == T[dep_stop] and it is better on tie-breakers:
            fewer transfers first, then less walking.

            This matches the README requirement: "all other things being equal,
            prefer routes with minimum walking distance and minimum number of transfers."

        Returns:
            (path, confidence, n_transfers, walk_m)
            Returns ([], 0.0, 0, 0.0) if no path exists.
        """
        NEG_INF = -1
        INF_INT = 10 ** 9

        T           = defaultdict(lambda: NEG_INF)   # stop -> latest feasible departure time
        C           = defaultdict(float)              # stop -> confidence
        N           = defaultdict(lambda: INF_INT)    # stop -> n_transfers
        W           = defaultdict(lambda: float('inf'))  # stop -> total walk_m
        predecessor = {}

        T[end_id] = deadline
        C[end_id] = 1.0
        N[end_id] = 0
        W[end_id] = 0.0
        self._propagate_walking_backward(end_id, deadline, T, C, N, W, predecessor,
                                         min_transfer_secs, max_walk_m)

        trip_reachable  = [False] * len(self.trip_ids)
        trip_confidence = [0.0]   * len(self.trip_ids)
        trip_transfers  = [0]     * len(self.trip_ids)  # STEP 4
        trip_walk_m     = [0.0]   * len(self.trip_ids)  # STEP 4

        for row in self.arr_sorted_list:
            arr_time = row[C_ARR_TIME]

            if T[start_id] != NEG_INF and arr_time < T[start_id]:
                break

            if not row[day_col]:
                continue

            dep_stop = row[C_DEP_STOP]
            arr_stop = row[C_ARR_STOP]
            dep_time = row[C_DEP_TIME]
            trip_idx = row[C_TRIP_IDX]

            if not trip_reachable[trip_idx]:
                if T[arr_stop] == NEG_INF:
                    continue
                transfer = 0 if arr_stop == end_id else min_transfer_secs
                if arr_time + transfer > T[arr_stop]:
                    continue

                slack_sec  = T[arr_stop] - arr_time - transfer
                trip_id    = self.trip_ids[trip_idx]
                product_id = self.trip_products.get(trip_id)
                line_text  = self.trip_lines.get(trip_id)

                
                # Option A: only penalise confidence when this is a genuine transfer or
                # final arrival. If arr_stop was already labelled by the same trip
                # propagating backwards, p_transfer = 1.0 (passenger stays on vehicle).
                arr_pred = predecessor.get(arr_stop)
                arr_pred_trip  = arr_pred[1] if arr_pred else None
                arr_pred_label = self._route_label(arr_pred_trip) if arr_pred_trip else None
                this_label     = self._route_label(trip_id)
                is_real_transfer = (
                    arr_stop == end_id                          or
                    arr_pred is None                            or
                    arr_pred_trip is None                       or   # arr_stop reached by walking
                    (arr_pred_label != this_label)                   # different LINE, not just different trip_id
                )

                p_transfer = (
                    self._p_transfer(arr_stop, arr_time, slack_sec, confidence_threshold,
                                     dow_bucket=dow_bucket,
                                     product_id=product_id, line_text=line_text)
                    if is_real_transfer else 1.0
                )


                c_new = p_transfer * C[arr_stop]

                if c_new < confidence_threshold:
                    continue

                # STEP 4: boarding a new trip costs one transfer (unless it's the
                # very first vehicle from the origin — handled by N[end_id]=0 seed).
                # A transfer is only counted when we board after having alighted,
                # i.e. when arr_stop already has a label from a different trip.
                # We conservatively count every new boarding as a transfer here;
                # the origin boarding is absorbed because N[end_id] starts at 0.
                n_new = N[arr_stop] + 1
                w_new = W[arr_stop]  # no extra walking for a vehicle boarding

            else:
                # Continuing on the same vehicle — inherit all criteria unchanged.
                c_new = trip_confidence[trip_idx]
                n_new = trip_transfers[trip_idx]
                w_new = trip_walk_m[trip_idx]

            # STEP 4 — dominance check replaces the simple `dep_time <= T[dep_stop]` guard.
            # A new label is accepted if:
            #   (a) it departs strictly later (always better), OR
            #   (b) it departs at the same time but has fewer transfers, OR
            #   (c) it departs at the same time, same transfers, but less walking.
            # reject labels at start_id that depart too late
            if max_dep_time is not None and dep_stop == start_id and dep_time >= max_dep_time:
                continue

            
            existing_time = T[dep_stop]
            if dep_time < existing_time:
                continue  # strictly worse on primary criterion — reject
            if dep_time == existing_time:
                if n_new > N[dep_stop]:
                    continue  # same time, more transfers — reject
                if n_new == N[dep_stop] and w_new >= W[dep_stop]:
                    continue  # same time, same transfers, no walking improvement — reject

            trip_id               = self.trip_ids[trip_idx]
            T[dep_stop]           = dep_time
            C[dep_stop]           = c_new
            N[dep_stop]           = n_new
            W[dep_stop]           = w_new
            predecessor[dep_stop] = (arr_stop, trip_id, dep_time, arr_time)
            trip_reachable[trip_idx]  = True
            trip_confidence[trip_idx] = c_new
            trip_transfers[trip_idx]  = n_new
            trip_walk_m[trip_idx]     = w_new

            self._propagate_walking_backward(dep_stop, dep_time, T, C, N, W, predecessor,
                                              min_transfer_secs, max_walk_m)

        if T[start_id] == NEG_INF:
            return [], 0.0, 0, 0.0

        path = self._reconstruct_backward_path(start_id, end_id, predecessor)
        if not path:
            return [], 0.0, 0, 0.0

        # N counts boardings from end_id outward; subtract 1 to get transfers
        # (transfers = boardings - 1, minimum 0).
        n_transfers = max(0, N[start_id] - 1)
        return path, C[start_id], n_transfers, W[start_id]

    # ------------------------------------------------------------------
    # STEP 3 — Unified transfer probability using DelayLookup
    # ------------------------------------------------------------------
    def _p_transfer(self, stop_id, arr_time_sec, slack_sec, confidence_threshold,
                    dow_bucket=DOW_BUCKET_WEEKDAY, product_id=None, line_text=None):
        """
        P(traveler catches the next connection) given slack_sec seconds of buffer
        at stop_id.

        Priority:
          1. DelayLookup (empirical CDFs, 6-level fallback hierarchy).
          2. Normal model from delay_features.
          3. Returns 1.0 when neither is loaded, or when confidence_threshold == 0.0.
        """
        if confidence_threshold == 0.0:
            return 1.0

        if self.delay_lookup is not None:
            hour = (arr_time_sec // 3600) % 24
            try:
                return self.delay_lookup.p_make_connection(
                    arr_bpuic=stop_id,
                    arr_hour=hour,
                    dep_bpuic=stop_id,
                    dep_hour=hour,
                    dow_bucket=dow_bucket,
                    slack_seconds=slack_sec,
                    arr_product=product_id,
                    dep_product=product_id,
                    arr_line=line_text,
                    dep_line=line_text,
                )
            except KeyError:
                pass

        if not self.delay_features:
            return 1.0

        hour  = (arr_time_sec // 3600) % 24
        stats = self.delay_features.get((stop_id, hour))
        if stats is None:
            return 1.0

        avg_sec, std_sec = stats
        if std_sec <= 0:
            return 1.0 if slack_sec >= avg_sec else 0.0

        return float(norm.cdf(slack_sec, loc=avg_sec, scale=std_sec))

    def _reconstruct_backward_path(self, start_id, end_id, predecessor):
        """
        Reconstruct path from backward CSA predecessor dict.
        predecessor[stop] = (next_stop, trip_id, dep_time, arr_time)

        Consecutive hops on the same trip_id are collapsed into a single
        board+alight pair so that through-rides on metro/bus/train lines are
        not displayed as repeated alight-and-reboard events.
        """
        if start_id not in predecessor:
            return []

        # 1. Build the raw hop chain
        hops    = []
        current = start_id
        visited = {start_id}

        while current != end_id:
            if current not in predecessor:
                break
            next_stop, trip_id, dep_time, arr_time = predecessor[current]
            if next_stop in visited:
                return []  # cycle — discard
            visited.add(next_stop)
            hops.append((current, next_stop, trip_id, dep_time, arr_time))
            current = next_stop

        if not hops:
            return []

        # 2. Collapse consecutive hops on the same trip into one board+alight pair
        events = []
        i = 0
        while i < len(hops):
            from_stop, to_stop, trip_id, dep_time, arr_time = hops[i]

            if trip_id is None:
                # walking hop — emit as-is
                events.append((dep_time, from_stop, None, to_stop, None))
                i += 1
            else:
                # scan forward while the trip_id is the same
                board_stop = from_stop
                board_time = dep_time
                alight_stop = to_stop
                alight_time = arr_time
                j = i + 1
                while j < len(hops) and (hops[j][2] == trip_id or self._route_label(hops[j][2]) == self._route_label(trip_id)):
                    alight_stop = hops[j][1]
                    alight_time = hops[j][4]
                    j += 1

                label = self._route_label(trip_id)
                events.append((board_time,  board_stop,  trip_id, None,        label))  # board
                events.append((alight_time, None,        trip_id, alight_stop, label))  # alight
                i = j

        return sorted(events, key=lambda x: x[0])

    def _would_create_cycle(self, neighbor, from_stop, predecessor):
        """Return True if setting predecessor[neighbor] = (from_stop, ...) would create a cycle."""
        current = from_stop
        seen = {neighbor}
        while current in predecessor:
            if current in seen:
                return True
            seen.add(current)
            current = predecessor[current][0]
        return False

    def _propagate_walking_backward(self, from_stop, from_time, T, C, N, W, predecessor,
                                    min_transfer_secs, max_walk_m):
        """
        Backward walking propagation.

        Walking edges carry no delay (P_walk = 1.0) and no transfer penalty,
        so C and N are inherited unchanged. W is increased by the walking distance.

        STEP 4: N and W parameters added so walking propagation carries all four
        label criteria. Signature is internal-only — no external callers.
        """
        for neighbor, dist_m in self.walking.get(from_stop, []):
            if dist_m > max_walk_m:
                continue
            walk_secs     = int(dist_m / WALK_SPEED_M_PER_SEC)
            t_at_neighbor = from_time - walk_secs
            w_at_neighbor = W[from_stop] + dist_m

            # Apply the same dominance logic as the main scan loop.
            existing_time = T[neighbor]
            if t_at_neighbor < existing_time:
                continue
            if t_at_neighbor == existing_time:
                if N[from_stop] > N[neighbor]:
                    continue
                if N[from_stop] == N[neighbor] and w_at_neighbor >= W[neighbor]:
                    continue

            T[neighbor] = t_at_neighbor
            C[neighbor] = C[from_stop]
            N[neighbor] = N[from_stop]   # walking doesn't add a transfer
            W[neighbor] = w_at_neighbor  # STEP 4: accumulate walk distance
            if not self._would_create_cycle(neighbor, from_stop, predecessor):
                predecessor[neighbor] = (from_stop, None, t_at_neighbor, from_time)

    def _confidence(self, path, min_transfer_secs, dow_bucket=DOW_BUCKET_WEEKDAY):
        """
        .. deprecated::
            Does not include the final leg arrival penalty.
            Use _confidence_with_final_leg() instead, which matches scan_confidence.
            Kept only for backward compatibility — do not call directly.
        """
        return self._confidence_with_final_leg(
            path,
            deadline=None,   # no final leg penalty
            min_transfer_secs=min_transfer_secs,
            dow_bucket=dow_bucket,
        )

    def _confidence_with_final_leg(self, path, deadline, min_transfer_secs,
                                   dow_bucket=DOW_BUCKET_WEEKDAY):
        """
        Like _confidence() but also penalises the final leg arrival.
        P(all transfers caught) × P(last vehicle arrives at end_id by deadline).
        This matches scan_confidence from _backward_csa.
        """
        if not self.delay_features and self.delay_lookup is None:
            return 1.0
    
        p_total          = 1.0
        last_alight_stop = None
        last_alight_time = None
        last_trip_id     = None
        final_alight_stop = None
        final_alight_time = None
    
        for ts, s1, trip_id, s2, *_ in path:
            if trip_id is not None and s2 is None:   # board
                if last_alight_time is not None and trip_id != last_trip_id:
                    # real transfer
                    slack_sec  = ts - last_alight_time
                    product_id = self.trip_products.get(trip_id)
                    line_text  = self.trip_lines.get(trip_id)
                    p_total   *= self._p_transfer(
                        last_alight_stop, last_alight_time, slack_sec,
                        confidence_threshold=1.0,
                        dow_bucket=dow_bucket,
                        product_id=product_id,
                        line_text=line_text,
                    )
            elif trip_id is not None and s1 is None:  # alight
                last_alight_stop  = s2
                last_alight_time  = ts
                last_trip_id      = trip_id
                final_alight_stop = s2
                final_alight_time = ts
    
        # Final leg: P(last vehicle arrives by deadline)
        if final_alight_stop is not None and final_alight_time is not None:
            slack_sec  = deadline - final_alight_time
            product_id = self.trip_products.get(last_trip_id)
            line_text  = self.trip_lines.get(last_trip_id)
            p_total   *= self._p_transfer(
                final_alight_stop, final_alight_time, slack_sec,
                confidence_threshold=1.0,
                dow_bucket=dow_bucket,
                product_id=product_id,
                line_text=line_text,
            )
    
        return p_total

    def explain_confidence(self, route_result, arrives_by=None, min_transfer_secs=120,
                           dow_bucket=DOW_BUCKET_WEEKDAY):
        """
        Print a per-transfer confidence breakdown for a route result dict
        (as returned by routes()).

        Compresses consecutive board/alight events of the same trip into one leg,
        and recomputes the true confidence regardless of the threshold used during
        the original search.

        Example usage:
            results = planner.routes(start_id, end_id, '09:00', 'tuesday',
                                     confidence_threshold=0.0)
            planner.explain_confidence(results[0])
        """
        fmt  = lambda s: f"{s // 3600:02d}:{(s % 3600) // 60:02d}"
        path = route_result['path']

        # Recompute true confidence (route_result['confidence'] may be 1.0 when
        # the search was run with confidence_threshold=0.0).
        # Recompute confidence the same way scan_confidence does:
        # product of p_transfer at each real transfer AND the final arrival.
        true_conf = route_result['confidence']
        # If the route was run at Q=0 (confidence=1.0), recompute properly
        # by re-calling _p_transfer at each transfer and at the final leg.
        if true_conf == 1.0 and arrives_by is not None:
            h, m = arrives_by.split(':')
            deadline = int(h) * 3600 + int(m) * 60
            true_conf = self._confidence_with_final_leg(path, deadline, min_transfer_secs, dow_bucket)
        print(f"Route: {fmt(route_result['dep_time'])} → {fmt(route_result['arr_time'])}  "
              f"(true confidence: {true_conf:.0%})")
        print()

        # Compress consecutive board/alight events of the same trip into one leg
        # so we only see trip boundaries (same logic as _compress_path in vis.py).
        legs  = []   # each leg: (board_ts, board_stop, trip_id, alight_ts, alight_stop) | walk
        i     = 0
        while i < len(path):
            ts, s1, trip_id, s2, route_label = path[i]
            if trip_id is not None and s2 is None:   # board
                board_ts, board_stop = ts, s1
                last_alight_ts, last_alight_stop = None, None
                j = i + 1
                while j < len(path):
                    ats, as1, atrip, as2, alabel = path[j]
                    if atrip == trip_id and as1 is None:   # alight same trip
                        last_alight_ts, last_alight_stop = ats, as2
                        j += 1
                        if j < len(path) and path[j][2] == trip_id and path[j][3] is None:
                            j += 1   # skip re-board at same stop
                            continue
                    break
                legs.append(('transit', board_ts, board_stop, trip_id,
                             last_alight_ts, last_alight_stop))
                i = j
            else:                                         # walk
                legs.append(('walk', ts, s1, s2))
                i += 1

        # Print legs and transfers
        p_running        = 1.0
        last_alight_stop = None
        last_alight_time = None
        last_trip_id     = None

        for leg in legs:
            if leg[0] == 'transit':
                _, board_ts, board_stop, trip_id, alight_ts, alight_stop = leg
                board_name  = self.stops.get(board_stop,  {}).get('name', board_stop)
                alight_name = self.stops.get(alight_stop, {}).get('name', alight_stop)
                line_text   = self.trip_lines.get(trip_id)

                if last_alight_time is not None and trip_id != last_trip_id:
                    prev_alight = self.stops.get(last_alight_stop, {}).get('name', last_alight_stop)
                    slack_sec   = board_ts - last_alight_time
                    p_catch     = self._p_transfer(
                        last_alight_stop, last_alight_time, slack_sec,
                        confidence_threshold=1.0,
                        dow_bucket=dow_bucket,
                        product_id=self.trip_products.get(trip_id),
                        line_text=line_text,
                    )
                    p_running *= p_catch
                    print(f"  ↔ Transfer: {prev_alight} {fmt(last_alight_time)}"
                          f" → {board_name} {fmt(board_ts)}"
                          f"  slack={slack_sec}s ({slack_sec//60}m{slack_sec%60:02d}s)"
                          f"  P={p_catch:.0%}  cumul={p_running:.0%}")

                tag = f"[{line_text}]" if line_text else f"[{trip_id}]"
                print(f"  🚌 {fmt(board_ts)} Board  {board_name} → "
                      f"Alight {alight_name} {fmt(alight_ts)}  {tag}")
                last_alight_stop = alight_stop
                last_alight_time = alight_ts
                last_trip_id     = trip_id

            else:   # walk
                _, ts, s1, s2 = leg
                n1 = self.stops.get(s1, {}).get('name', s1)
                n2 = self.stops.get(s2, {}).get('name', s2)
                print(f"  🚶 {fmt(ts)} Walk   {n1} → {n2}")

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------
    def debug_route(self, start_id, end_id, arrives_by, day,
                    confidence_threshold=0.8, min_transfer_secs=120,
                    max_walk_m=500, watch_stops=()):
        """
        Run the backward CSA with verbose tracing for a handful of stops.

        watch_stops: iterable of stop_ids whose T-value changes you want logged.
        """
        from collections import defaultdict

        NEG_INF = -1
        h, m = arrives_by.split(':')
        deadline = int(h) * 3600 + int(m) * 60
        day_col  = C_DAY_BASE + DAY_COLUMNS.index(day.lower())
        dow_bucket = _DAY_TO_DOW_BUCKET[day.lower()]

        def fmt(s): return f"{s//3600:02d}:{(s%3600)//60:02d}"

        # ---- 1. Check walking edges from destination ----
        print("=== Walking edges from destination ===")
        dest_name = self.stops.get(end_id, {}).get('name', end_id)
        print(f"Destination: {dest_name} (id={end_id})")
        walk_from_dest = self.walking.get(end_id, [])
        if not walk_from_dest:
            print("  !! NO walking edges from destination — T[Castolin] will never be seeded!")
        for nb, dist in sorted(walk_from_dest, key=lambda x: x[1]):
            nb_name = self.stops.get(nb, {}).get('name', nb)
            print(f"  → {nb_name} (id={nb})  {dist:.0f}m")

        # ---- 2. Init T and walking propagation ----
        T           = defaultdict(lambda: NEG_INF)
        C           = defaultdict(float)
        N           = defaultdict(lambda: 10**9)
        W           = defaultdict(lambda: float('inf'))
        predecessor = {}

        T[end_id] = deadline; C[end_id] = 1.0; N[end_id] = 0; W[end_id] = 0.0
        self._propagate_walking_backward(end_id, deadline, T, C, N, W, predecessor,
                                         min_transfer_secs, max_walk_m)

        print("\n=== T-values after initial walking propagation ===")
        for sid, tv in sorted(T.items(), key=lambda x: -x[1]):
            if tv == NEG_INF: continue
            name = self.stops.get(sid, {}).get('name', sid)
            print(f"  T[{name}] = {fmt(tv)}")

        # ---- 3. Count how many connections land at each watched stop ----
        watch_set = set(watch_stops) | {end_id}
        print(f"\n=== Connections in arr_sorted_list landing at watched stops ===")
        for sid in watch_set:
            name = self.stops.get(sid, {}).get('name', sid)
            hits = [(r[C_ARR_TIME], r[C_DEP_STOP], r[C_DEP_TIME])
                    for r in self.arr_sorted_list
                    if r[C_ARR_STOP] == sid and r[day_col]]
            print(f"  arr_stop={name}: {len(hits)} connections on {day}")
            for at, ds, dt in sorted(hits, key=lambda x: -x[0])[:5]:
                ds_name = self.stops.get(ds, {}).get('name', ds)
                print(f"    from {ds_name} dep={fmt(dt)} arr={fmt(at)}")
            if len(hits) > 5:
                print(f"    ... and {len(hits)-5} more")

        # ---- 4. Full scan with logging for watched stops ----
        trip_reachable  = [False] * len(self.trip_ids)
        trip_confidence = [0.0]   * len(self.trip_ids)
        trip_transfers  = [0]     * len(self.trip_ids)
        trip_walk_m     = [0.0]   * len(self.trip_ids)

        print(f"\n=== CSA scan (logging events for watched stops) ===")
        break_reason = "exhausted"
        for row in self.arr_sorted_list:
            arr_time = row[C_ARR_TIME]

            if T[start_id] != NEG_INF and arr_time < T[start_id]:
                break_reason = f"early-exit: arr_time={fmt(arr_time)} < T[start]={fmt(T[start_id])}"
                break

            if not row[day_col]:
                continue

            dep_stop = row[C_DEP_STOP]
            arr_stop = row[C_ARR_STOP]
            dep_time = row[C_DEP_TIME]
            trip_idx = row[C_TRIP_IDX]

            in_watch = (dep_stop in watch_set or arr_stop in watch_set)

            if not trip_reachable[trip_idx]:
                if T[arr_stop] == NEG_INF:
                    if in_watch:
                        ds_n = self.stops.get(dep_stop,{}).get('name',dep_stop)
                        as_n = self.stops.get(arr_stop,{}).get('name',arr_stop)
                        print(f"  SKIP (T[{as_n}]=NEG_INF): {ds_n}→{as_n} dep={fmt(dep_time)} arr={fmt(arr_time)}")
                    continue
                transfer = 0 if arr_stop == end_id else min_transfer_secs
                if arr_time + transfer > T[arr_stop]:
                    if in_watch:
                        ds_n = self.stops.get(dep_stop,{}).get('name',dep_stop)
                        as_n = self.stops.get(arr_stop,{}).get('name',arr_stop)
                        print(f"  SKIP (time): {ds_n}→{as_n} dep={fmt(dep_time)} arr={fmt(arr_time)} "
                              f"arr+tr={fmt(arr_time+transfer)} > T[{as_n}]={fmt(T[arr_stop])}")
                    continue
                slack_sec  = T[arr_stop] - arr_time - transfer
                trip_id    = self.trip_ids[trip_idx]
                product_id = self.trip_products.get(trip_id)
                line_text  = self.trip_lines.get(trip_id)
                arr_pred = predecessor.get(arr_stop)
                is_real_transfer = (
                    arr_stop == end_id or
                    arr_pred is None or
                    arr_pred[1] is None or
                    arr_pred[1] != trip_id
                )
                p_transfer = (
                    self._p_transfer(arr_stop, arr_time, slack_sec, confidence_threshold,
                                     dow_bucket=dow_bucket, product_id=product_id, line_text=line_text)
                    if is_real_transfer else 1.0
                )
                c_new = p_transfer * C[arr_stop]

                if c_new < confidence_threshold:
                    if in_watch:
                        ds_n = self.stops.get(dep_stop,{}).get('name',dep_stop)
                        as_n = self.stops.get(arr_stop,{}).get('name',arr_stop)
                        print(f"  SKIP (conf {c_new:.0%}<{confidence_threshold:.0%}): "
                              f"{ds_n}→{as_n} dep={fmt(dep_time)} arr={fmt(arr_time)}")
                    continue

                n_new = N[arr_stop] + 1
                w_new = W[arr_stop]
            else:
                c_new = trip_confidence[trip_idx]
                n_new = trip_transfers[trip_idx]
                w_new = trip_walk_m[trip_idx]

            existing_time = T[dep_stop]
            if dep_time < existing_time:
                continue
            if dep_time == existing_time:
                if n_new > N[dep_stop]: continue
                if n_new == N[dep_stop] and w_new >= W[dep_stop]: continue

            trip_id = self.trip_ids[trip_idx]
            old_t   = T[dep_stop]
            T[dep_stop]           = dep_time
            C[dep_stop]           = c_new
            N[dep_stop]           = n_new
            W[dep_stop]           = w_new
            predecessor[dep_stop] = (arr_stop, trip_id, dep_time, arr_time)
            trip_reachable[trip_idx]  = True
            trip_confidence[trip_idx] = c_new
            trip_transfers[trip_idx]  = n_new
            trip_walk_m[trip_idx]     = w_new

            if in_watch or dep_stop in watch_set:
                ds_n = self.stops.get(dep_stop,{}).get('name',dep_stop)
                as_n = self.stops.get(arr_stop,{}).get('name',arr_stop)
                print(f"  ACCEPT: {ds_n}→{as_n} dep={fmt(dep_time)} arr={fmt(arr_time)} "
                      f"conf={c_new:.0%}  T[{ds_n}]: {fmt(old_t) if old_t!=NEG_INF else 'NEG_INF'}→{fmt(dep_time)}")

            self._propagate_walking_backward(dep_stop, dep_time, T, C, N, W, predecessor,
                                             min_transfer_secs, max_walk_m)
            if dep_stop in watch_set:
                for nb, dist in self.walking.get(dep_stop, []):
                    if T[nb] != NEG_INF:
                        nb_n = self.stops.get(nb,{}).get('name',nb)
                        print(f"    walk→ {nb_n} T={fmt(T[nb])}")

        print(f"\nScan ended: {break_reason}")
        sn = self.stops.get(start_id,{}).get('name',start_id)
        print(f"T[{sn}] = {fmt(T[start_id]) if T[start_id]!=NEG_INF else 'NEG_INF'}")
