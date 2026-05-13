# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
"""
vis.py

Callable visualization and demo functions.
"""


def launch_demo(planner):
    """
    Main function called by demo_app.py.

    TODO:
    - Create ipywidgets controls:
        start stop
        end stop
        day
        latest arrival time
        confidence Q
        max walking distance
    - On button click, call planner.routes(...)
    - Display map and route summaries.
    """
    raise NotImplementedError


def render_route_summary(routes):
    """
    Display route options as readable text/table.

    Temporary version can just print.
    Later version can use rich notebook display.
    """
    if not routes:
        print("No robust route found.")
        return

    for i, route in enumerate(routes, start=1):

        ## TODO: formatting route information


def build_route_map(routes, stops):
    """
    Build map visualization.

    TODO:
    - Use folium, ipyleaflet, or plotly.
    - Draw stop markers.
    - Draw lines for each leg.
    - Use different styling for walking vs transit.
    """
    raise NotImplementedError


def render_route_details(route):
    """
    Render expandable SBB-style route details.

    Each leg should show:
    - mode
    - from/to stops
    - departure/arrival time
    - trip/line
    - confidence contribution
    """
    raise NotImplementedError
# -

# ## Map Requirements:
#
# ### Inputs:
# - start location
# - end location
# - date/time of departure or arrival
# - (optional) max desired walking distance
#   
# ### Behaviors:
# (on start)
# - map of Switzerland
# - fields for start and end stops (maybe has a search feature or region-select in order to quickly sort)
# - field for customized date/time (auto filled with current day + time)
#
# (upon user-input)
# - display of suggested fastest routes (on the map)
# - display of suggested fastest route information + arrival time confidences (in text below)
#
# (additional)
# - expandable-route interface (like on the SBB app), where the user can see each mode of transport, stop-over times, etc. for a specific route option
# - 
