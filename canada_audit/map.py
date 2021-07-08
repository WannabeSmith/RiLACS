import pandas as pd
import geopandas as gpd
import numpy as np
from bokeh.layouts import row, column
from bokeh.plotting import figure, curdoc
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    WheelZoomTool,
    TapTool,
    CategoricalColorMapper,
    LegendItem,
    Select,
)
from bokeh.events import Tap
from functools import partial
from tornado import gen
from threading import Thread

import sys
import os

sys.path.append(os.path.relpath("../"))
from rilacs.incremental_audit import *

canada_map = gpd.GeoDataFrame.from_file("Data/map_digital.shp")


def getPolyCoords(row, geom, coord_type):
    """
    Returns the coordinates ('x' or 'y') of edges of a Polygon exterior.
    This is is just required to get things into a format Bokeh understands.
    """
    exterior = row[geom].exterior
    if coord_type == "x":
        return list(exterior.coords.xy[0])
    elif coord_type == "y":
        return list(exterior.coords.xy[1])


# Get Bokeh-understandable x and y coords.
canada_map["x"] = canada_map.apply(
    getPolyCoords, geom="geometry", coord_type="x", axis=1
)
canada_map["y"] = canada_map.apply(
    getPolyCoords, geom="geometry", coord_type="y", axis=1
)

can_df = canada_map.drop("geometry", axis=1).copy()
can_df = can_df[["FEDNUM", "ENNAME", "PROVCODE", "x", "y"]]
can_df = can_df.rename(
    columns={
        "FEDNUM": "District number",
        "ENNAME": "District name",
        "PROVCODE": "Province",
    }
)

# Fix riding 24020's name as it conflicts with the votes dataset
can_df.loc[
    can_df["District number"] == 24020, "District name"
] = "Beauport-Côte-de-Beaupré-Île d'Orléans-Charlevoix"

# Merge the two data sets
votes_df = pd.read_csv("Data/Canada_2021_federal_election_results.csv")
electoral_map_df = pd.merge(
    can_df, votes_df, how="left", on=["District number", "District name"]
)
# Ensure that 'Winner_shorthand' is stored as a string. This will be
# essential when using legend_group in map_figure.patches
# https://stackoverflow.com/questions/46406720/labelencoder-typeerror-not-supported-between-instances-of-float-and-str
electoral_map_df["Winner_shorthand"] = (
    electoral_map_df["Winner_shorthand"].astype(str).fillna(0)
)

can_source = ColumnDataSource(electoral_map_df)

colors = {
    "Bloc": "cyan",
    "Green": "green",
    "PPC": "purple",
    "NDP": "orange",
    "PC": "blue",
    "Liberal": "red",
    "Independent": "grey",
    "Other": "white",
}

parties = [party for party in list(colors.keys()) if party != "Other"]


def unzip(l):
    return zip(*l)


factors, palette = unzip(
    [(party, color) for party, color in colors.items() if party != "Other"]
)
cmap = CategoricalColorMapper(palette=palette, factors=factors)

# Configure map plot
map_figure = figure(
    title="Canadian federal electoral districts", sizing_mode="scale_both"
)
map_figure.patches(
    "x",
    "y",
    source=can_source,
    fill_color={"field": "Winner_shorthand", "transform": cmap},
    fill_alpha=0.5,
    line_color="grey",
    line_width=0.8,
    legend_group="Winner_shorthand",
)
map_figure.axis.visible = False
map_figure.title.text_font_size = "15pt"
map_figure.legend.label_text_font_size = "15pt"

# When we hover over a riding, want a popup which says its name
tooltip = HoverTool()
tooltip.tooltips = [
    ("Riding", "@{District name}"),
    ("Liberal", "@{Liberal}"),
    ("PC", "@{Conservative}"),
    ("Green", "@{Green Party}"),
    ("NDP", "@{NDP-New Democratic Party}"),
    ("PPC", "@{People's Party}"),
    ("Bloc", "@{Bloc Québécois}"),
    ("Independent", "@{Independent}"),
    ("Total Votes", "@{Total votes}"),
]

# Want to be able to zoom in by scrolling
map_figure.toolbar.active_scroll = map_figure.select_one(WheelZoomTool)

taptool = TapTool()
map_figure.add_tools(tooltip, taptool)

audit_figure = figure(title="Audit", sizing_mode="scale_both")

select = Select(
    title="Audit method",
    value="Kelly",
    options=["Kelly", "SqKelly", "Hoeffding"],
    width=120,
    sizing_mode="fixed",
)

doc = curdoc()
doc.title = "Canada Election Audit"


party_plot = {party: None for party in parties}
for party in parties:
    source = ColumnDataSource(data={"t": [], "l": []})
    party_plot[party] = {
        "source": source,
        "line": audit_figure.line(
            x="t",
            y="l",
            source=source,
            color=colors[party],
            legend_label=party,
            line_width=4,
            alpha=0.5,
        ),
    }
audit_figure.xaxis.axis_label = "Ballots sampled"
audit_figure.yaxis.axis_label = "Lower confidence sequence for mean difference"
audit_figure.axis.axis_label_text_font_size = "20pt"
audit_figure.axis.major_label_text_font_size = "15pt"
audit_figure.title.text_font_size = "15pt"
audit_figure.legend.label_text_font_size = "15pt"

state = np.random.get_state()


@gen.coroutine
def reset_plot_sources():
    for party in parties:
        party_plot[party]["source"].data = {"t": [], "l": []}


@gen.coroutine
def update_plot_title(title):
    audit_figure.title.text = title


@gen.coroutine
def update_plot_legend(reported_winner):
    party_renderers = {party: party_plot[party]["line"] for party in parties}
    audit_figure.legend.items = [
        LegendItem(label=party, renderers=[r])
        for party, r in party_renderers.items()
        if party != reported_winner
    ]


@gen.coroutine
def update_plot_outline_color(reported_winner):
    audit_figure.outline_line_width = 7
    audit_figure.outline_line_alpha = 0.3
    audit_figure.outline_line_color = colors[reported_winner]


@gen.coroutine
def update_party_plot_source(new_data):
    for party in parties:
        # https://stackoverflow.com/questions/24800071/timeseries-streaming-in-bokeh/37185420#37185420
        party_plot[party]["source"].stream(new_data[party])


def perform_audit():
    selected_idx = can_source.selected.indices[0]
    district_num = electoral_map_df["District number"][selected_idx]
    district_name = electoral_map_df["District name"][selected_idx]
    province = electoral_map_df["Province"][selected_idx]

    district_data = electoral_map_df.iloc[[selected_idx]]

    # Create a dictionary of all the votes for the most common parties.
    votes = {
        "Liberal": int(list(district_data["Liberal"])[0]),
        "PC": int(list(district_data["Conservative"])[0]),
        "NDP": int(list(district_data["NDP-New Democratic Party"])[0]),
        "Green": int(list(district_data["Green Party"])[0]),
        "PPC": int(list(district_data["People's Party"])[0]),
        "Bloc": int(list(district_data["Bloc Québécois"])[0]),
        "Independent": int(list(district_data["Independent"])[0]),
        "Other": int(list(district_data["Other votes"])[0]),
    }

    reported_winner = max(votes, key=lambda key: votes[key])
    competitors = [party for party in parties if party != reported_winner]

    doc.add_next_tick_callback(
        partial(
            update_plot_title,
            title="Audit of "
            + district_name
            + ", "
            + province
            + ". Reported winner: "
            + reported_winner
            + ".",
        )
    )
    doc.add_next_tick_callback(
        partial(update_plot_legend, reported_winner=reported_winner)
    )
    doc.add_next_tick_callback(
        partial(update_plot_outline_color, reported_winner=reported_winner)
    )

    # Create a list of votes. e.g. ['Liberal', 'PC', 'Liberal', 'Green', ...]
    ballots = np.concatenate(
        [[party] * num_votes for party, num_votes in votes.items()]
    )

    np.random.set_state(state)
    np.random.shuffle(ballots)

    ballot_dict = {}
    audits = {}
    audit_methods_dict = {
        "Kelly": lambda n_A, n_B: Betting_Audit(
            bettor=Kelly_Bettor(n_A=n_A, n_B=n_B), N=len(ballots)
        ),
        "SqKelly": lambda n_A, n_B: Betting_Audit(
            bettor=DistKelly_Bettor(), N=len(ballots)
        ),
        "Hoeffding": lambda n_A, n_B: Hoeffding_Audit(N=len(ballots)),
    }

    for competitor in competitors:
        competitor_dict = {party: 0.5 for party in parties}
        competitor_dict[reported_winner] = 1
        competitor_dict[competitor] = 0

        # Map reported winner to 1, competitor to 0,
        # and everything else to 0.5 using competitor_dict
        ballot_dict[competitor] = np.array(list(map(competitor_dict.get, ballots)))

        n_A = sum(ballot_dict[competitor] == 1)
        n_B = sum(ballot_dict[competitor] == 0)

        selected_audit_method = select.value
        audits[competitor] = audit_methods_dict[selected_audit_method](n_A=n_A, n_B=n_B)
    print("Starting audit of " + district_name)
    new_data = {party: {"t": [], "l": []} for party in parties}
    for i in range(len(ballots)):
        ballot = ballots[i]

        if min([audits[competitor].l for competitor in competitors]) > 0.5:
            print("Stopping time = " + str(i + 1))
            break
        for competitor in competitors:
            if audits[competitor].l <= 0.5:
                # Encode, winner: 1, competitor: 0, anything else: 0.5
                ballot_num = (
                    1
                    if ballot == reported_winner
                    else 0
                    if ballot == competitor
                    else 0.5
                )

                l = audits[competitor].update_cs(ballot_num)
                new_data[competitor]["t"].append(audits[competitor].t)
                new_data[competitor]["l"].append(l)
        # Only update plots every few ballots. Otherwise, bokeh gets overwhelmed.
        if i % 100 == 0:
            doc.add_next_tick_callback(partial(update_party_plot_source, new_data))
            new_data = {party: {"t": [], "l": []} for party in parties}

    # Update plots with leftover ballots
    doc.add_next_tick_callback(partial(update_party_plot_source, new_data))
    new_data = {party: {"t": [], "l": []} for party in parties}
    print("Audit complete")


def riding_click_callback(event):
    global state
    state = np.random.get_state()
    doc.add_next_tick_callback(reset_plot_sources)
    # https://docs.bokeh.org/en/latest/docs/user_guide/server.html#updating-from-threads
    thread = Thread(target=perform_audit)
    thread.start()


def audit_method_select_callback(attr, old, new):
    doc.add_next_tick_callback(reset_plot_sources)
    thread = Thread(target=perform_audit)
    thread.start()


# Configure election audit plot
map_figure.on_event(Tap, riding_click_callback)
doc.add_root(
    column(
        select,
        row([map_figure, audit_figure], sizing_mode="scale_both"),
        sizing_mode="scale_both",
    )
)

select.on_change("value", audit_method_select_callback)
