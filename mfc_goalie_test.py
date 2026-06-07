from pathlib import Path
import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.font_manager import FontProperties
from mplsoccer import PyPizza


st.set_page_config(page_title="MFC Goalie Test", layout="wide")

APP_ROOT = Path(__file__).resolve().parent
MINUTE_THRESHOLD = 360

LEAGUE_DICT = {
    "ENG1": "Premier League",
    "ENG2": "Championship",
    "ENG3": "League One",
    "ENG4": "League Two",
    "SCO1": "Scottish Premiership",
    "AUS1": "A-League",
    "BRA1": "Brasileirao",
    "BEL1": "Pro League",
    "DEN1": "Superliga",
    "EGY1": "Egyptian Premier League",
    "GRE1": "Greek Superleague",
    "GER1": "Bundesliga",
    "GER2": "2. Bundesliga",
    "ITA1": "Serie A",
    "POR1": "Liga Portugal",
    "FRA1": "Ligue 1",
    "SWE1": "Allsvenskan",
    "NOR1": "Eliteserien",
    "AUT1": "Austrian Bundesliga",
    "IRE1": "League of Ireland",
    "JAP1": "J-League",
    "USA1": "MLS",
    "SOU1": "South African Premier",
    "SAU1": "Saudi Pro League",
    "TUR1": "Superlig",
}

SEASON_DICT = {
    "2223": "2022/23",
    "2324": "2023/24",
    "2425": "2024/25",
    "2526": "2025/26",
    "2025": "2025",
    "2026": "2026",
}


def league_label(code):
    return LEAGUE_DICT.get(str(code), str(code))


def season_label(code):
    return SEASON_DICT.get(str(code), str(code))


def discover_data_files(root):
    match_files = {}
    player_files = {}

    for path in root.rglob("*.parquet"):
        parts = path.stem.rsplit("_", 1)
        if len(parts) == 2:
            match_files[tuple(parts)] = path

    for path in root.rglob("*.xlsx"):
        if not path.stem.endswith("_playertotal"):
            continue
        parts = path.stem.removesuffix("_playertotal").rsplit("_", 1)
        if len(parts) == 2:
            player_files[tuple(parts)] = path

    return [
        {
            "league": league,
            "season": season,
            "matchlog": match_files[(league, season)],
            "playerlog": player_files[(league, season)],
        }
        for league, season in sorted(match_files.keys() & player_files.keys())
    ]


def ensure_columns(frame, columns, default=0):
    for col in columns:
        if col not in frame.columns:
            frame[col] = default
    return frame


def safe_divide(numerator, denominator):
    return np.where(pd.Series(denominator).gt(0), numerator / denominator, np.nan)


@st.cache_data(show_spinner=False, max_entries=1, ttl=600)
def load_source_data(matchlog, playerlog):
    return pd.read_parquet(matchlog), pd.read_excel(playerlog)


@st.cache_data(show_spinner="Building goalkeeper stats...", max_entries=1, ttl=600)
def build_goalie_stats(matchlog, playerlog, minute_threshold):
    df, player_totals = load_source_data(matchlog, playerlog)
    df = df.copy()
    player_totals = player_totals.copy()

    for col in ["expectedGoalsOnTarget", "expectedGoals", "xT_value", "x", "y", "end_x", "end_y"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    gk_events = df[df["playing_position"].eq("GK")].copy()
    is_pass = gk_events["typeId"].eq("Pass")

    pass_locations = [
        is_pass & (gk_events["x"] <= 5.8) & gk_events["y"].between(36.8, 63.2, inclusive="both"),
        is_pass & (gk_events["x"] <= 17) & gk_events["y"].between(21.1, 79.9, inclusive="both"),
        is_pass & (gk_events["x"] <= 17),
        is_pass & gk_events["x"].between(17, 33.3, inclusive="both"),
        is_pass & gk_events["x"].between(33.3, 66.6, inclusive="both"),
    ]
    pass_location_names = ["Six Yard Box", "Penalty Area", "Wide of Box", "Own Third", "Middle Third"]
    gk_events["pass_location"] = np.select(pass_locations, pass_location_names, default=None)

    pass_location_counts = (
        gk_events[gk_events["pass_location"].notna()]
        .groupby(["playerName", "pass_location"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=pass_location_names, fill_value=0)
        .rename(
            columns={
                "Six Yard Box": "six_yard_box_passes",
                "Penalty Area": "penalty_area_passes",
                "Wide of Box": "wide_of_box_passes",
                "Own Third": "own_third_passes",
                "Middle Third": "middle_third_passes",
            }
        )
    )

    attempted_passes = gk_events[is_pass].groupby("playerName").size().rename("attempted_passes")
    successful_passes = gk_events[is_pass & gk_events["outcome"].eq("Successful")].copy()
    completed_passes = successful_passes.groupby("playerName").size().rename("completed_passes")
    successful_passes["true_distance"] = np.sqrt(
        ((successful_passes["end_x"] - successful_passes["x"]) / 100 * 105) ** 2
        + ((successful_passes["end_y"] - successful_passes["y"]) / 100 * 68) ** 2
    )

    passes_to_cb = successful_passes[successful_passes["pass_recipient_position"].str.contains("CB", na=False)].groupby("playerName").size().rename("passes_to_cb")
    passes_to_fb = successful_passes[successful_passes["pass_recipient_position"].str.contains("LB|RB|LWB|RWB", na=False)].groupby("playerName").size().rename("passes_to_fb")
    passes_to_cm = successful_passes[successful_passes["pass_recipient_position"].str.contains("CM|DM", na=False)].groupby("playerName").size().rename("passes_to_cm")
    pass_15 = successful_passes[successful_passes["true_distance"].le(15)].groupby("playerName").size().rename("pass_15")
    pass_15to30 = successful_passes[successful_passes["true_distance"].gt(15) & successful_passes["true_distance"].le(30)].groupby("playerName").size().rename("pass_15to30")
    pass_30to45 = successful_passes[successful_passes["true_distance"].gt(30) & successful_passes["true_distance"].le(45)].groupby("playerName").size().rename("pass_30to45")
    pass_45_plus = successful_passes[successful_passes["true_distance"].gt(45)].groupby("playerName").size().rename("pass_45_plus")

    successful_passes["end_location"] = np.select(
        [
            successful_passes["end_y"].between(0, 20, inclusive="both"),
            successful_passes["end_y"].gt(20) & successful_passes["end_y"].le(40),
            successful_passes["end_y"].gt(40) & successful_passes["end_y"].le(60),
            successful_passes["end_y"].gt(60) & successful_passes["end_y"].le(80),
            successful_passes["end_y"].gt(80) & successful_passes["end_y"].le(100),
        ],
        ["Wide Right", "Centre Right", "Centre", "Centre Left", "Wide Left"],
        default=None,
    )
    end_location_counts = (
        successful_passes[successful_passes["end_location"].notna()]
        .groupby(["playerName", "end_location"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Wide Right", "Centre Right", "Centre", "Centre Left", "Wide Left"], fill_value=0)
        .rename(
            columns={
                "Wide Right": "passes_end_wide_right",
                "Centre Right": "passes_end_centre_right",
                "Centre": "passes_end_centre",
                "Centre Left": "passes_end_centre_left",
                "Wide Left": "passes_end_wide_left",
            }
        )
    )

    passing_threat = gk_events[is_pass & gk_events["outcome"].eq("Successful")].groupby("playerName")["xT_value"].sum().rename("passing_threat")
    attempted_claim_punch = gk_events[gk_events["typeId"].isin(["Claim", "Punch"])].groupby("playerName").size().rename("attempted_claim_punch")
    keeper_sweeper = gk_events[gk_events["typeId"].isin(["Keeper Sweeper"])].groupby("playerName").size().rename("keeper_sweeper")
    successful_claim_punch = gk_events[gk_events["typeId"].isin(["Claim", "Punch"]) & gk_events["outcome"].eq("Successful")].groupby("playerName").size().rename("successful_claim_punch")
    errors = gk_events[gk_events["typeId"].eq("Error")].groupby("playerName").size().rename("errors")

    player_name_stats = (
        pd.concat(
            [
                attempted_passes,
                completed_passes,
                passes_to_cb,
                passes_to_fb,
                passes_to_cm,
                pass_15,
                pass_15to30,
                pass_30to45,
                pass_45_plus,
                passing_threat,
                attempted_claim_punch,
                keeper_sweeper,
                successful_claim_punch,
                errors,
                pass_location_counts,
                end_location_counts,
            ],
            axis=1,
        )
        .rename_axis("playerName")
        .reset_index()
    )

    count_cols = [
        "attempted_passes",
        "completed_passes",
        "passes_to_cb",
        "passes_to_fb",
        "passes_to_cm",
        "pass_15",
        "pass_15to30",
        "pass_30to45",
        "pass_45_plus",
        "passing_threat",
        "attempted_claim_punch",
        "successful_claim_punch",
        "errors",
        "keeper_sweeper",
        "six_yard_box_passes",
        "penalty_area_passes",
        "wide_of_box_passes",
        "own_third_passes",
        "middle_third_passes",
        "passes_end_wide_right",
        "passes_end_centre_right",
        "passes_end_centre",
        "passes_end_centre_left",
        "passes_end_wide_left",
    ]
    player_name_stats = ensure_columns(player_name_stats, count_cols)
    player_name_stats[count_cols] = player_name_stats[count_cols].fillna(0)

    for col in ["six_yard_box_passes", "penalty_area_passes", "wide_of_box_passes", "own_third_passes", "middle_third_passes"]:
        player_name_stats[f"{col}_pct"] = safe_divide(player_name_stats[col], player_name_stats["attempted_passes"])
    for col in ["passes_to_cb", "passes_to_fb", "passes_to_cm"]:
        player_name_stats[f"{col}_pct"] = safe_divide(player_name_stats[col], player_name_stats["completed_passes"])
    for col in ["pass_15", "pass_15to30", "pass_30to45", "pass_45_plus"]:
        player_name_stats[f"{col}_pct"] = safe_divide(player_name_stats[col], player_name_stats["completed_passes"])
    for col in ["passes_end_wide_right", "passes_end_centre_right", "passes_end_centre", "passes_end_centre_left", "passes_end_wide_left"]:
        player_name_stats[f"{col}_pct"] = safe_divide(player_name_stats[col], player_name_stats["completed_passes"])

    player_name_stats["passing_threat_per_10_passes"] = safe_divide(player_name_stats["passing_threat"], player_name_stats["attempted_passes"]) * 10
    player_name_stats["pass_completion"] = safe_divide(player_name_stats["completed_passes"], player_name_stats["attempted_passes"])
    player_name_stats["claim_punch_success"] = safe_divide(player_name_stats["successful_claim_punch"], player_name_stats["attempted_claim_punch"])
    player_name_stats = player_name_stats.drop(columns=["completed_passes", "successful_claim_punch"])

    passes_to_gk = df[df["typeId"].eq("Pass") & df["pass_recipient_position"].eq("GK")].copy()
    is_pass_to_gk = passes_to_gk["typeId"].eq("Pass")
    reception_conditions = [
        is_pass_to_gk & (passes_to_gk["end_x"] <= 5.8) & passes_to_gk["end_y"].between(36.8, 63.2, inclusive="both"),
        is_pass_to_gk & (passes_to_gk["end_x"] <= 17) & passes_to_gk["end_y"].between(21.1, 79.9, inclusive="both"),
        is_pass_to_gk & (passes_to_gk["end_x"] <= 17),
        is_pass_to_gk & passes_to_gk["end_x"].between(17, 33.3, inclusive="both"),
        is_pass_to_gk & passes_to_gk["end_x"].between(33.3, 66.6, inclusive="both"),
    ]
    passes_to_gk["pass_reception_location"] = np.select(reception_conditions, pass_location_names, default=None)

    passes_received = passes_to_gk.groupby("pass_recipient").size().rename("passes_received").reset_index().rename(columns={"pass_recipient": "playerName"})
    average_pass_reception_height = passes_to_gk.groupby("pass_recipient")["end_x"].mean().rename("average_pass_reception_height").reset_index().rename(columns={"pass_recipient": "playerName"})
    pass_reception_location_counts = (
        passes_to_gk[passes_to_gk["pass_reception_location"].notna()]
        .groupby(["pass_recipient", "pass_reception_location"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=pass_location_names, fill_value=0)
        .rename(
            columns={
                "Six Yard Box": "six_yard_box_receptions",
                "Penalty Area": "penalty_area_receptions",
                "Wide of Box": "wide_of_box_receptions",
                "Own Third": "own_third_receptions",
                "Middle Third": "middle_third_receptions",
            }
        )
        .reset_index()
        .rename(columns={"pass_recipient": "playerName"})
    )
    passes_received = passes_received.merge(average_pass_reception_height, on="playerName", how="left")
    passes_received = passes_received.merge(pass_reception_location_counts, on="playerName", how="left")
    reception_count_cols = ["six_yard_box_receptions", "penalty_area_receptions", "wide_of_box_receptions", "own_third_receptions", "middle_third_receptions"]
    passes_received = ensure_columns(passes_received, reception_count_cols)
    passes_received[reception_count_cols] = passes_received[reception_count_cols].fillna(0)
    for col in reception_count_cols:
        passes_received[f"{col}_pct"] = safe_divide(passes_received[col], passes_received["passes_received"])

    playing_gk_stats = (
        df.groupby("playing_GK")
        .agg(
            gettable_crosses=("cross_into_six", lambda s: s.str.lower().eq("yes").sum()),
            expected_goals_on_target=("expectedGoalsOnTarget", "sum"),
            shots_on_target=("expectedGoalsOnTarget", "count"),
            goals_conceded=("typeId", lambda s: s.eq("Goal").sum()),
            big_chances_faced=("expectedGoals", lambda s: (s.ge(0.3) & df.loc[s.index, "expectedGoalsOnTarget"].gt(0)).sum()),
            big_chances_saved=("expectedGoals", lambda s: (s.ge(0.3) & df.loc[s.index, "expectedGoalsOnTarget"].gt(0) & df.loc[s.index, "typeId"].ne("Goal")).sum()),
        )
        .reset_index()
        .rename(columns={"playing_GK": "playerName"})
    )

    goalie_stats = player_name_stats.merge(passes_received, on="playerName", how="left").merge(playing_gk_stats, on="playerName", how="left").fillna(0)
    player_totals = player_totals[player_totals["playing_position"].str.contains("GKGKGKGK", na=False)]
    minutes_played = player_totals[["player_name", "minutes_played"]].drop_duplicates(subset=["player_name"]).rename(columns={"player_name": "playerName"})
    goalie_stats = goalie_stats.merge(minutes_played, on="playerName", how="left")
    goalie_stats["minutes_played"] = pd.to_numeric(goalie_stats["minutes_played"], errors="coerce").fillna(0)

    goalie_stats["attempted_passes_per_90"] = safe_divide(goalie_stats["attempted_passes"], goalie_stats["minutes_played"]) * 90
    goalie_stats["passes_received_per_90"] = safe_divide(goalie_stats["passes_received"], goalie_stats["minutes_played"]) * 90
    goalie_stats["goals_conceded_per_90"] = safe_divide(goalie_stats["goals_conceded"], goalie_stats["minutes_played"]) * 90
    goalie_stats["goals_prevented"] = goalie_stats["expected_goals_on_target"] - goalie_stats["goals_conceded"]
    goalie_stats["goals_prevented_per_90"] = safe_divide(goalie_stats["goals_prevented"], goalie_stats["minutes_played"]) * 90
    goalie_stats["cross_intervention_rate"] = safe_divide(goalie_stats["attempted_claim_punch"], goalie_stats["gettable_crosses"])
    goalie_stats["true_save%"] = 1 - safe_divide(goalie_stats["goals_conceded"], goalie_stats["shots_on_target"])
    goalie_stats["big_chance_save%"] = safe_divide(goalie_stats["big_chances_saved"], goalie_stats["big_chances_faced"])
    goalie_stats["errors_per_90"] = safe_divide(goalie_stats["errors"], goalie_stats["minutes_played"]) * 90
    goalie_stats["keeper_sweeper_per_90"] = safe_divide(goalie_stats["keeper_sweeper"], goalie_stats["minutes_played"]) * 90
    goalie_stats = goalie_stats.replace([np.inf, -np.inf], np.nan)
    goalie_stats = goalie_stats[goalie_stats["minutes_played"] >= minute_threshold].reset_index(drop=True)

    percentile_cols = [
        "six_yard_box_passes_pct",
        "penalty_area_passes_pct",
        "wide_of_box_passes_pct",
        "own_third_passes_pct",
        "middle_third_passes_pct",
        "passes_to_cb_pct",
        "passes_to_fb_pct",
        "passes_to_cm_pct",
        "pass_15_pct",
        "pass_15to30_pct",
        "pass_30to45_pct",
        "pass_45_plus_pct",
        "passes_end_wide_right_pct",
        "passes_end_centre_right_pct",
        "passes_end_centre_pct",
        "passes_end_centre_left_pct",
        "passes_end_wide_left_pct",
        "passing_threat_per_10_passes",
        "pass_completion",
        "claim_punch_success",
        "average_pass_reception_height",
        "six_yard_box_receptions_pct",
        "penalty_area_receptions_pct",
        "wide_of_box_receptions_pct",
        "own_third_receptions_pct",
        "middle_third_receptions_pct",
        "attempted_passes_per_90",
        "passes_received_per_90",
        "goals_conceded_per_90",
        "goals_prevented_per_90",
        "cross_intervention_rate",
        "true_save%",
        "big_chance_save%",
        "errors_per_90",
        "keeper_sweeper_per_90",
    ]
    inverted_cols = {"errors_per_90", "goals_conceded_per_90"}
    for col in percentile_cols:
        goalie_stats[f"{col}_percentile"] = goalie_stats[col].rank(pct=True, ascending=col not in inverted_cols) * 100

    return goalie_stats


def make_pizza(goalie_stats, player_name, title, pizza_cols, params, slice_colors, text_colors, league_name, season_name):
    player_row = goalie_stats.loc[goalie_stats["playerName"].eq(player_name)].iloc[0]
    values = player_row[pizza_cols].fillna(0).round(0).astype(int).tolist()
    font_normal = FontProperties()
    font_bold = FontProperties(weight="bold")
    font_italic = FontProperties(style="italic")

    baker = PyPizza(
        params=params,
        background_color="#F2F2F2",
        straight_line_color="#F2F2F2",
        straight_line_lw=1,
        last_circle_lw=0,
        other_circle_lw=0,
        inner_circle_size=20,
    )
    fig, _ = baker.make_pizza(
        values,
        figsize=(6.8, 7.2),
        color_blank_space="same",
        slice_colors=slice_colors,
        value_colors=text_colors,
        value_bck_colors=slice_colors,
        blank_alpha=0.4,
        kwargs_slices={"edgecolor": "#F2F2F2", "zorder": 2, "linewidth": 1},
        kwargs_params={"color": "#000000", "fontsize": 10, "fontproperties": font_normal, "va": "center"},
        kwargs_values={
            "color": "#000000",
            "fontsize": 10,
            "fontproperties": font_normal,
            "zorder": 3,
            "bbox": {"edgecolor": "#000000", "facecolor": "cornflowerblue", "boxstyle": "round,pad=0.2", "lw": 1},
        },
    )
    fig.text(0.515, 0.975, f"{player_name} - GK Percentile Rank (0-100) - {title}", size=14, ha="center", fontproperties=font_bold)
    fig.text(0.515, 0.953, f"Compared against other {league_name} goalkeepers | {season_name}", size=12, ha="center", fontproperties=font_bold)
    fig.text(0.05, 0.02, f"Data from Opta | Metrics are per 90 unless stated | Minimum {MINUTE_THRESHOLD} mins played", size=9, fontproperties=font_italic, ha="left")
    return fig


CHARTS = [
    {
        "tab": "Keeping",
        "title": "Keeping",
        "pizza_cols": [
            "keeper_sweeper_per_90_percentile",
            "claim_punch_success_percentile",
            "cross_intervention_rate_percentile",
            "errors_per_90_percentile",
            "goals_conceded_per_90_percentile",
            "goals_prevented_per_90_percentile",
            "true_save%_percentile",
            "big_chance_save%_percentile",
        ],
        "params": ["Keeper Sweeper", "Claim/Punch\nSuccess", "Cross Intervention\nRate", "Errors", "Goals Conceded", "Goals Prevented", "True Save %", "Big Chance Save %"],
        "slice_colors": ["red"] * 4 + ["#63ace3"] * 4,
        "text_colors": ["#000000"] * 8,
    },
    {
        "tab": "Passing",
        "title": "Passing",
        "pizza_cols": [
            "attempted_passes_per_90_percentile",
            "pass_completion_percentile",
            "passing_threat_per_10_passes",
            "pass_15_pct_percentile",
            "pass_15to30_pct_percentile",
            "pass_30to45_pct_percentile",
            "pass_45_plus_pct_percentile",
            "passes_to_cb_pct_percentile",
            "passes_to_fb_pct_percentile",
            "passes_to_cm_pct_percentile",
            "passes_end_wide_right_pct_percentile",
            "passes_end_centre_right_pct_percentile",
            "passes_end_centre_pct_percentile",
            "passes_end_centre_left_pct_percentile",
            "passes_end_wide_left_pct_percentile",
        ],
        "params": ["Attempted Passes", "Pass\nCompletion %", "Passing\nxThreat", "Passes\nCompleted\nShort", "Passes\nCompleted\n15-30yds", "Passes\nCompleted\n30-45yds", "Passes\nCompleted\nLong", "Passes\nto CBs", "Passes\nto FBs", "Passes\nto CMs", "Passes to\nWide Right", "Passes to\nCentre Right", "Passes to\nCentre", "Passes to\nCentre Left", "Passes to\nWide Left"],
        "slice_colors": ["red"] * 7 + ["#63ace3"] * 3 + ["#2f316a"] * 5,
        "text_colors": ["#000000"] * 10 + ["white"] * 5,
    },
    {
        "tab": "Receiving",
        "title": "Receiving",
        "pizza_cols": [
            "passes_received_per_90_percentile",
            "average_pass_reception_height_percentile",
            "six_yard_box_receptions_pct_percentile",
            "penalty_area_receptions_pct_percentile",
            "wide_of_box_receptions_pct_percentile",
            "own_third_receptions_pct_percentile",
            "middle_third_receptions_pct_percentile",
        ],
        "params": ["Passes Received", "Average Reception\nHeight", "Received in\n6 Yard Box", "Received in\nPenalty Box", "Received\nWide of Box", "Received in\nOwn Third", "Received in\nMiddle Third"],
        "slice_colors": ["red"] * 2 + ["#63ace3"] * 5,
        "text_colors": ["#000000"] * 7,
    },
]


st.title("MFC Goalie Test")

available_files = discover_data_files(APP_ROOT)
if not available_files:
    st.error("No matching files found. Upload files named like `DEN1_2526.parquet` and `DEN1_2526_playertotal.xlsx`.")
    st.stop()

league_options = sorted({item["league"] for item in available_files}, key=league_label)
league = st.selectbox("League", league_options, format_func=lambda code: f"{league_label(code)} ({code})")

season_options = sorted({item["season"] for item in available_files if item["league"] == league}, key=lambda code: (season_label(code), code))
season = st.selectbox("Season", season_options, format_func=lambda code: f"{season_label(code)} ({code})")

selected_files = next(item for item in available_files if item["league"] == league and item["season"] == season)
goalie_stats = build_goalie_stats(str(selected_files["matchlog"]), str(selected_files["playerlog"]), MINUTE_THRESHOLD)

if goalie_stats.empty:
    st.warning(f"No goalkeepers met the {MINUTE_THRESHOLD} minute threshold for {league_label(league)} {season_label(season)}.")
    st.stop()

player_name = st.selectbox("Goalkeeper", goalie_stats["playerName"].dropna().sort_values().tolist())

for tab, chart in zip(st.tabs([chart["tab"] for chart in CHARTS]), CHARTS):
    with tab:
        fig = make_pizza(
            goalie_stats,
            player_name,
            chart["title"],
            chart["pizza_cols"],
            chart["params"],
            chart["slice_colors"],
            chart["text_colors"],
            league_label(league),
            season_label(season),
        )
        left, centre, right = st.columns([1, 3, 1])
        with centre:
            st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        del fig
        gc.collect()
