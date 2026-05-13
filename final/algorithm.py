#!/usr/bin/env python
# coding: utf-8
# %%
'''
algorithm.py
CSA routing algorithm + route confidence calculations
'''


# %%


######################## pasted from Assignment 1 - com490.py

import numpy as np
from collections import defaultdict
from scipy.stats import norm


DAY_COLUMNS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
WALK_SPEED_M_PER_SEC = 50 / 60  # assignment spec: 50m per minute

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
        self.stops         = {}   # stop_id (int) -> {name, lat, lon}
        self.walking       = {}   # stop_id -> [(neighbor_id, distance_m)]
        self.trip_ids      = []   # list of trip_id strings, indexed by C_TRIP_IDX
        self.delay_features = {}  # (stop_id, hour_of_day) -> (avg_delay_sec, std_delay_sec)

        # core CSA data structures
        self.conn_arr   = None   # np.ndarray (N,12) sorted by dep_time for forward CSA
        self.dep_times  = None   # 1-D view of C_DEP_TIME
        self.arr_sorted = None   # np.ndarray (N,12) sorted by arr_time desc for backward CSA

    def prepare_from_data(self, data):
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

        day_cols = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        self.conn_arr = conn_pd[
            ['dep_stop_id', 'dep_time_sec', 'arr_stop_id', 'arr_time_sec', 'trip_idx'] + day_cols
        ].astype('int64').values
        self.dep_times  = self.conn_arr[:, C_DEP_TIME]
        self.arr_sorted = self.conn_arr[np.argsort(self.conn_arr[:, C_ARR_TIME])[::-1]]
        print(f"Loaded {len(self.conn_arr)} connections")

        # walking edges
        for row in data.walking_edges.collect():
            self.walking.setdefault(int(row['a_stop_id']), []).append(
                (int(row['b_stop_id']), float(row['distance_m']))
            )
        print(f"Loaded walking edges for {len(self.walking)} stops")

        # historical delay features — optional, only present when istdaten was prepared
        if data.delay_features is not None:
            for row in data.delay_features.collect():
                key = (int(row['bpuic']), int(row['hour_of_day']))
                self.delay_features[key] = (
                    float(row['hist_avg_delay_mins']) * 60,  # convert to seconds
                    float(row['hist_std_delay_mins']) * 60,
                )
            print(f"Loaded delay features for {len(self.delay_features)} (stop, hour) pairs")
        else:
            print("No delay features available — confidence will default to 1.0")

    def describe_route(self, route_result):
        """Print a single routes() result in human-readable form."""
        dep  = route_result['dep_time']
        arr  = route_result['arr_time']
        conf = route_result['confidence']
        fmt  = lambda s: f"{s // 3600:02d}:{(s % 3600) // 60:02d}"
        print(f"Departure {fmt(dep)} → Arrival {fmt(arr)}  (confidence {conf:.0%})")
        for ts, s1, trip_id, s2 in route_result['path']:
            t = fmt(ts)
            if trip_id and s2 is None:
                print(f"  {t}  board  {trip_id}  at {self.stops.get(s1, {}).get('name', s1)}")
            elif trip_id and s1 is None:
                print(f"  {t}  alight at {self.stops.get(s2, {}).get('name', s2)}")
            else:
                n1 = self.stops.get(s1, {}).get('name', s1)
                n2 = self.stops.get(s2, {}).get('name', s2)
                print(f"  {t}  walk   {n1} → {n2}")


    def routes(self, start_id, end_id, arrives_by, day,
               confidence_threshold=0.9, max_routes=5,
               min_transfer_secs=120, max_walk_m=500):
        """
        Find up to max_routes routes from start_id to end_id arriving before arrives_by.

        Returns a list of dicts sorted from latest departure to earliest:
            [{'path': [...], 'confidence': float, 'dep_time': int, 'arr_time': int}, ...]
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

        h, m = arrives_by.split(':')
        deadline = int(h) * 3600 + int(m) * 60
        day_col  = C_DAY_BASE + DAY_COLUMNS.index(day.lower())

        results          = []
        current_deadline = deadline

        for _ in range(max_routes):
            path = self._backward_csa(
                start_id, end_id, current_deadline, day_col,
                min_transfer_secs, max_walk_m,
            )
            if not path:
                break

            # Simplified confidence — replace with delay model later
            confidence = self._confidence(path, min_transfer_secs)
            if confidence < confidence_threshold:
                break

            dep_time = path[0][0]
            arr_time = path[-1][0]
            results.append({
                'path':       path,
                'confidence': confidence,
                'dep_time':   dep_time,
                'arr_time':   arr_time,
            })

            current_deadline = dep_time - 1  # next route must depart strictly earlier

        return results

    def _backward_csa(self, start_id, end_id, deadline, day_col,
                      min_transfer_secs, max_walk_m):
        """
        Backward CSA: find the latest-departing path from start_id reaching end_id by deadline.

        T[stop] = latest time we can board/depart from stop and still reach end_id in time.
        """
        NEG_INF = -1
        T           = defaultdict(lambda: NEG_INF)
        predecessor = {}

        T[end_id] = deadline
        self._propagate_walking_backward(end_id, deadline, T, predecessor,
                                         min_transfer_secs, max_walk_m)

        # trip_reachable[trip_idx] = True once a later segment of this trip has
        # been accepted — no transfer buffer needed to continue on the same vehicle
        trip_reachable = [False] * len(self.trip_ids)

        for row in self.arr_sorted.tolist():
            arr_time = row[C_ARR_TIME]

            # Early termination: any remaining connection has dep_time <= arr_time < T[start_id],
            # so it cannot improve T[start_id] directly or through propagation.
            if T[start_id] != NEG_INF and arr_time < T[start_id]:
                break

            if not row[day_col]:
                continue

            dep_stop = row[C_DEP_STOP]
            arr_stop = row[C_ARR_STOP]
            dep_time = row[C_DEP_TIME]
            trip_idx = row[C_TRIP_IDX]

            # Can we use this connection to reach end_id?
            # Already on this trip → traveler stays on the vehicle, no transfer needed
            # Boarding for the first time → arr_stop must be reachable; at intermediate
            #   stops min_transfer_secs must be available, but not at end_id itself
            if not trip_reachable[trip_idx]:
                if T[arr_stop] == NEG_INF:
                    continue
                transfer = 0 if arr_stop == end_id else min_transfer_secs
                if arr_time + transfer > T[arr_stop]:
                    continue

            # Does this connection let us depart later from dep_stop?
            if dep_time <= T[dep_stop]:
                continue

            trip_id               = self.trip_ids[trip_idx]
            T[dep_stop]           = dep_time
            predecessor[dep_stop] = (arr_stop, trip_id, dep_time, arr_time)
            trip_reachable[trip_idx] = True

            self._propagate_walking_backward(dep_stop, dep_time, T, predecessor,
                                              min_transfer_secs, max_walk_m)

        if T[start_id] == NEG_INF:
            return []

        return self._reconstruct_backward_path(start_id, end_id, predecessor)

    def _reconstruct_backward_path(self, start_id, end_id, predecessor):
        """
        Reconstruct path from backward CSA predecessor dict.
        predecessor[stop] = (next_stop, trip_id, dep_time, arr_time)
        so we follow forward from start_id to end_id.
        """
        if start_id not in predecessor:
            print(f"No path found from stop {start_id}")
            return []

        events  = []
        current = start_id

        while current != end_id:
            if current not in predecessor:
                break
            next_stop, trip_id, dep_time, arr_time = predecessor[current]

            if trip_id is not None:
                events.append((dep_time, current,   trip_id, None))      # board
                events.append((arr_time, None,      trip_id, next_stop)) # alight
            else:
                events.append((arr_time, current,   None,    next_stop)) # walk

            current = next_stop

        return sorted(events, key=lambda x: x[0])

    def _propagate_walking_backward(self, from_stop, from_time, T, predecessor,
                                    min_transfer_secs, max_walk_m):
        """
        Backward walking: if we can depart from_stop at from_time,
        we can also depart a walking-neighbor earlier by walk_secs.
        """
        for neighbor, dist_m in self.walking.get(from_stop, []):
            if dist_m > max_walk_m:
                continue
            walk_secs      = int(dist_m / WALK_SPEED_M_PER_SEC)
            t_at_neighbor  = from_time - walk_secs
            if t_at_neighbor > T[neighbor]:
                T[neighbor]           = t_at_neighbor
                predecessor[neighbor] = (from_stop, None, t_at_neighbor, from_time)

    def _confidence(self, path, min_transfer_secs):
        """
        Probability that the traveler catches every transfer on this path.

        For each real transfer (alight trip A, board trip B), compute:
            P(delay of A at alight_stop <= slack)
        where slack = dep_time_B - arr_time_A and delay is modelled as
        N(avg_delay_sec, std_delay_sec) from historical Istdaten.

        Returns 1.0 when no delay features are loaded.
        The overall confidence is the product of per-transfer probabilities.
        """
        if not self.delay_features:
            return 1.0

        p_total          = 1.0
        last_alight_stop = None
        last_alight_time = None
        last_trip_id     = None

        for ts, s1, trip_id, s2 in path:
            if trip_id is not None and s2 is None:  # board event
                if last_alight_time is not None and trip_id != last_trip_id:
                    # Real transfer: compute P(catch connection)
                    slack_sec = ts - last_alight_time
                    hour      = (last_alight_time // 3600) % 24
                    stats     = self.delay_features.get((last_alight_stop, hour))
                    if stats is not None:
                        avg_sec, std_sec = stats
                        if std_sec > 0:
                            p_catch = float(norm.cdf(slack_sec, loc=avg_sec, scale=std_sec))
                        else:
                            p_catch = 1.0 if slack_sec >= avg_sec else 0.0
                        p_total *= p_catch

            elif trip_id is not None and s1 is None:  # alight event
                last_alight_stop = s2
                last_alight_time = ts
                last_trip_id     = trip_id

        return p_total
