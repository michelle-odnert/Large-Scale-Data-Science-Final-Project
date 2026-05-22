"""
vis.py

Interactive demo UI using ipywidgets.
Called by demo_app.py via launch_demo(planner).
"""

import folium
import ipywidgets as w
from IPython.display import display, HTML


def launch_demo(planner):
    """Interactive journey planner widget, called by run_demo()."""

    stop_names = sorted(
        {info["name"]: sid for sid, info in planner.stops.items()}.items()
    )
    name_to_id = dict(stop_names)
    names = [n for n, _ in stop_names]

    default_origin = "Lausanne" if "Lausanne" in name_to_id else names[0]
    default_dest   = "Renens VD" if "Renens VD" in name_to_id else names[1]

    origin = w.Combobox(
        value=default_origin, options=names,
        description="From", ensure_option=True,
        layout=w.Layout(width="320px"),
    )
    dest = w.Combobox(
        value=default_dest, options=names,
        description="To", ensure_option=True,
        layout=w.Layout(width="320px"),
    )
    arrives_by = w.Text(
        value="09:00", description="Arrive by",
        layout=w.Layout(width="160px"),
    )
    day = w.Dropdown(
        options=["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
        value="monday", description="Day",
        layout=w.Layout(width="200px"),
    )
    confidence = w.FloatSlider(
        value=0.80, min=0.0, max=0.99, step=0.05,
        description="Confidence", readout_format=".0%",
        style={"description_width": "initial"},
        layout=w.Layout(width="360px"),
    )
    max_routes = w.BoundedIntText(
        value=3, min=1, max=5, description="Max routes",
        layout=w.Layout(width="160px"),
    )

    btn = w.Button(
        description="Search", button_style="primary",
        icon="search", layout=w.Layout(width="140px"),
    )
    map_out = w.Output()
    routes_out = w.Output()
    status_out = w.Output()

    def _fmt(secs):
        return f"{secs // 3600:02d}:{(secs % 3600) // 60:02d}"

    def _compress_path(path):
        """Merge consecutive board/alight pairs of the same trip into one leg."""
        compressed = []
        i = 0
        while i < len(path):
            ts, s1, trip_id, s2, route_label = path[i]
            if trip_id and s2 is None:  # board event — find where this trip ends
                board_ts, board_stop = ts, s1
                j = i + 1
                last_alight_ts, last_alight_stop = None, None
                while j < len(path):
                    ats, as1, atrip, as2, alabel = path[j]
                    if atrip == trip_id and as1 is None:  # alight same trip
                        last_alight_ts, last_alight_stop = ats, as2
                        j += 1
                        # peek: if next is a board on the same trip, keep merging
                        if j < len(path) and path[j][2] == trip_id and path[j][3] is None:
                            j += 1  # skip the re-board at the same stop
                            continue
                    break
                compressed.append((board_ts, board_stop, trip_id, None, route_label))
                if last_alight_ts is not None:
                    compressed.append((last_alight_ts, None, trip_id, last_alight_stop, route_label))
                i = j
            else:
                compressed.append((ts, s1, trip_id, s2, route_label))
                i += 1
        return compressed

    def _route_endpoint_stop_ids(route):
        path = route["path"]
    
        start_sid = None
        dest_sid = None
    
        # First non-empty stop in the path
        for ts, s1, trip_id, s2, route_label in path:
            if s1 is not None:
                start_sid = s1
                break
            if s2 is not None:
                start_sid = s2
                break
    
        # Last non-empty stop in the path
        for ts, s1, trip_id, s2, route_label in reversed(path):
            if s2 is not None:
                dest_sid = s2
                break
            if s1 is not None:
                dest_sid = s1
                break
    
        return start_sid, dest_sid

    def _render_route(r, idx):
        dep  = _fmt(r["dep_time"])
        arr  = _fmt(r["arr_time"])
        dur  = (r["arr_time"] - r["dep_time"]) // 60
        conf = r["confidence"]
        n_tr = r["n_transfers"]
        walk = r["walk_m"]

        lines = []
        
        for ts, s1, trip_id, s2, route_label in _compress_path(r["path"]):
            t = _fmt(ts)
            if trip_id and s2 is None:
                stop_name = planner.stops.get(s1, {}).get("name", str(s1))
                lines.append(
                    f"<div>🟢 <code>{t}</code> &nbsp; Board <b>{route_label}</b> at <b>{stop_name}</b>"
                    f" &nbsp;"
                )
            elif trip_id and s1 is None:
                stop_name = planner.stops.get(s2, {}).get("name", str(s2))
                lines.append(
                    f"<div>🔴 <code>{t}</code> &nbsp; Alight at <b>{stop_name}</b></div>"
                )
            else:
                n1 = planner.stops.get(s1, {}).get("name", str(s1))
                n2 = planner.stops.get(s2, {}).get("name", str(s2))
                lines.append(
                    f"<div>🚶 <code>{t}</code> &nbsp; Walk: {n1} → {n2}</div>"
                )

        lines.append("</div>")
        return "".join(lines)

    # Route colours for up to 5 options
    _ROUTE_COLORS = ["#1565c0", "#2e7d32", "#6a1b9a", "#e65100", "#37474f"]

    def _get_stop_coords(sid):
        info = planner.stops.get(sid, {})
        if info.get("lat") and info.get("lon"):
            return info["lat"], info["lon"], info.get("name", str(sid))
        return None

    def _can_draw_transfer(s, start_sid, dest_sid, seen_sids, trip_id, prev_trip_id):
        return s not in (start_sid, dest_sid) and s not in seen_sids and trip_id != prev_trip_id

    def _draw_transfer_circle(m, ts, lat, lon, name, r_idx):
        folium.CircleMarker(
            location=[lat, lon],
            radius=7,
            color="#f57c00",
            fill=True,
            fill_color="#f57c00",
            fill_opacity=0.95,
            popup=folium.Popup(
                f"<b>Transfer</b><br>{name}<br>{_fmt(ts)}",
                max_width=180,
            ),
            tooltip=f"Option {r_idx + 1} Transfer: {name}",
        ).add_to(m)

    def _build_map(routes):
        all_coords = []
        for r in routes:
            for ts, s1, trip_id, s2, route_label in r["path"]:
                for sid in (s1, s2):
                    if sid is not None:
                        info = planner.stops.get(sid, {})
                        if info.get("lat") and info.get("lon"):
                            all_coords.append((info["lat"], info["lon"]))

        if not all_coords:
            return None

        center_lat = sum(c[0] for c in all_coords) / len(all_coords)
        center_lon = sum(c[1] for c in all_coords) / len(all_coords)
        m = folium.Map(location=[center_lat, center_lon], zoom_start=13,
                       tiles="CartoDB positron")
        
        start_sid, dest_sid = _route_endpoint_stop_ids(routes[0])
        seen_transfer_sids = set()
        
        for r_idx, r in enumerate(routes):
            color = _ROUTE_COLORS[r_idx % len(_ROUTE_COLORS)]
            path = r["path"]
        
            # Draw legs and transfer dots
            current_leg = []
            prev_trip_id = None
            for ts, s1, trip_id, s2, route_label in path:
                if trip_id and s2 is None:   # board
                    stop_info = _get_stop_coords(s1)
        
                    if stop_info:
                        lat, lon, name = stop_info
                        current_leg = [(lat, lon)]
        
                        # Draw orange transfer dot for board stops,
                        # except overall start and final destination
                        # must be at a change of transportation (where trip_ids not equal)
                        if _can_draw_transfer(s1, start_sid, dest_sid, seen_transfer_sids, trip_id, prev_trip_id):
                            _draw_transfer_circle(m, ts, lat, lon, name, r_idx)
        
                            seen_transfer_sids.add(s1)
        
                elif trip_id and s1 is None:  # alight
                    stop_info = _get_stop_coords(s2)
        
                    if stop_info:
                        lat, lon, name = stop_info
                        current_leg.append((lat, lon))
        
                        if len(current_leg) >= 2:
                            folium.PolyLine(
                                current_leg,
                                color=color,
                                weight=4,
                                opacity=0.8,
                                tooltip=f"Option {r_idx + 1}: {route_label}",
                            ).add_to(m)
        
                        # Draw orange transfer dot for alight stops,
                        # except overall start and final destination
                        if _can_draw_transfer(s2, start_sid, dest_sid, seen_transfer_sids, trip_id, prev_trip_id):
                            _draw_transfer_circle(m, ts, lat, lon, name, r_idx)
        
                            seen_transfer_sids.add(s2)
        
                        current_leg = []
        
                else:  # walk
                    stop1_info = _get_stop_coords(s1)
                    stop2_info = _get_stop_coords(s2)
        
                    if stop1_info and stop2_info:
                        lat1, lon1, name1 = stop1_info
                        lat2, lon2, name2 = stop2_info
        
                        folium.PolyLine(
                            [(lat1, lon1), (lat2, lon2)],
                            color=color,
                            weight=2,
                            opacity=0.7,
                            dash_array="6 4",
                            tooltip=f"Option {r_idx + 1} walking",
                        ).add_to(m)

                    if _can_draw_transfer(s1, start_sid, dest_sid, seen_transfer_sids, trip_id, prev_trip_id):
                        _draw_transfer_circle(m, ts, lat1, lon1, name1, r_idx)
    
                        seen_transfer_sids.add(s1)
                        
                    if _can_draw_transfer(s2, start_sid, dest_sid, seen_transfer_sids, trip_id, prev_trip_id):
                        _draw_transfer_circle(m, ts, lat2, lon2, name2, r_idx)
    
                        seen_transfer_sids.add(s2)

                prev_trip_id = trip_id

        # Draw only the main start and destination circles
        start_info = _get_stop_coords(start_sid)
        dest_info = _get_stop_coords(dest_sid)

        
        if start_info:
            lat, lon, name = start_info
            folium.CircleMarker(
                location=[lat, lon],
                radius=9,
                color="#2e7d32",
                fill=True,
                fill_color="#2e7d32",
                fill_opacity=0.95,
                popup=folium.Popup(f"<b>Start</b><br>{name}", max_width=180),
                tooltip=f"Start: {name}",
            ).add_to(m)
        
        if dest_info:
            lat, lon, name = dest_info
            folium.CircleMarker(
                location=[lat, lon],
                radius=9,
                color="#c62828",
                fill=True,
                fill_color="#c62828",
                fill_opacity=0.95,
                popup=folium.Popup(f"<b>Destination</b><br>{name}", max_width=180),
                tooltip=f"Destination: {name}",
            ).add_to(m)

        legend_items = ""
        
        legend_items += """
        <div style="margin-bottom:6px;">
            <span style="
                display:inline-block;
                width:12px;
                height:12px;
                background:#2e7d32;
                border-radius:50%;
                margin-right:6px;
            "></span>
            Start
        </div>
        """

        legend_items += """
        <div style="margin-bottom:6px;">
            <span style="
                display:inline-block;
                width:12px;
                height:12px;
                background:#f57c00;
                border-radius:50%;
                margin-right:6px;
            "></span>
            Transfer
        </div>
        """
        
        legend_items += """
        <div style="margin-bottom:6px;">
            <span style="
                display:inline-block;
                width:12px;
                height:12px;
                background:#c62828;
                border-radius:50%;
                margin-right:6px;
            "></span>
            Destination
        </div>
        """
        
        for idx, r in enumerate(routes, 1):
            color = _ROUTE_COLORS[(idx - 1) % len(_ROUTE_COLORS)]
            legend_items += f"""
            <div style="margin-bottom:6px;">
                <span style="
                    display:inline-block;
                    width:22px;
                    height:4px;
                    background:{color};
                    margin-right:6px;
                    vertical-align:middle;
                "></span>
                Option {idx}
            </div>
            """
        
        legend_html = f"""
        <div style="
            position: fixed;
            bottom: 30px;
            left: 30px;
            z-index: 9999;
            background: white;
            padding: 12px 14px;
            border: 1px solid #bbb;
            border-radius: 6px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.25);
            font-size: 13px;
        ">
            <div style="font-weight:bold; margin-bottom:8px;">Map key</div>
            {legend_items}
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        
        m.fit_bounds([[min(c[0] for c in all_coords) - 0.002,
                       min(c[1] for c in all_coords) - 0.002],
                      [max(c[0] for c in all_coords) + 0.002,
                       max(c[1] for c in all_coords) + 0.002]])
        return m

    def on_click(_):
    
        map_out.clear_output(wait=True)
        routes_out.clear_output(wait=True)
        status_out.clear_output(wait=True)
    
        with status_out:

            if origin.value not in name_to_id:
                display(HTML("<p style='color:red'>Unknown origin stop.</p>"))
                return
            if dest.value not in name_to_id:
                display(HTML("<p style='color:red'>Unknown destination stop.</p>"))
                return
            if origin.value == dest.value:
                display(HTML("<p style='color:orange'>Origin and destination are the same.</p>"))
                return

        btn.disabled = True
        btn.description = "Searching..."
        
        try:
            routes = planner.routes(
                name_to_id[origin.value],
                name_to_id[dest.value],
                arrives_by=arrives_by.value,
                day=day.value,
                confidence_threshold=confidence.value,
                max_routes=max_routes.value,
            )
        except Exception as e:
            with status_out:
                display(HTML(f"<p style='color:red'>Error: {e}</p>"))
            routes = []
        finally:
            btn.disabled = False
            btn.description = "Search"
    
        if not routes:
            with status_out:
                display(HTML(
                    f"<p style='color:orange'>No route found with "
                    f"{confidence.value:.0%} confidence.</p>"
                ))
            return

        # Display map on the right
        route_map = _build_map(routes)
        if route_map:
            with map_out:
                display(HTML(route_map._repr_html_()))
        
        # Then display routes underneath as dropdown panels
        route_widgets = []
        
        for i, r in enumerate(routes, 1):
            route_html = HTML(_render_route(r, i))
            route_widgets.append(w.Output())
        
            with route_widgets[-1]:
                display(route_html)
        
        accordion = w.Accordion(children=route_widgets)
        
        for i, r in enumerate(routes, 1):
            dep  = _fmt(r["dep_time"])
            arr  = _fmt(r["arr_time"])
            dur  = (r["arr_time"] - r["dep_time"]) // 60
            conf = r["confidence"]
            n_tr = r["n_transfers"]
            walk = r["walk_m"]
    
            accordion.set_title(
                i - 1,
                f"Option {i} | {dep} → {arr} "
                f"({dur} min, {n_tr} transfer(s), {walk:.0f} m walking) "
                f"| {conf:.0%} confidence"
            )

        accordion.selected_index = 0

        with routes_out:
            display(HTML(f"<p><b>{len(routes)} route(s) found</b> — latest departure first</p>"))
            display(accordion)

    btn.on_click(on_click)

    controls = w.VBox(
        [
            origin,
            dest,
            arrives_by,
            day,
            confidence,
            max_routes,
            btn,
            status_out,
        ],
        layout=w.Layout(
            width="380px",
            padding="12px",
            border="1px solid #ddd",
            margin="0 0 12px 0",
        ),
    )
    
    left_panel = w.VBox(
        [
            controls,
            routes_out,
        ],
        layout=w.Layout(
            width="420px",
            min_width="420px",
            margin="0 12px 0 0",
        ),
    )
    
    main = w.VBox(
        [map_out],
        layout=w.Layout(
            flex="1",
            min_width="600px",
        ),
    )
    
    ui = w.HBox(
        [left_panel, main],
        layout=w.Layout(
            width="100%",
            align_items="flex-start",
        ),
    )
    
    display(ui)


