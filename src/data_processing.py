import os
import pandas as pd


def load_and_prepare_data():
    """
    Loads, cleans, and combines data from multiple seasons into a single DataFrame.
    """
    # Get the directory of the current script
    current_dir = os.path.dirname(__file__)
    data_dir = os.path.join(current_dir, "../data")


    season_files = [
        "curry_2021-22_reg_season.csv",
        "curry_2022-23_reg_season.csv",
        "curry_2023-24_reg_season.csv",
        "curry_2024-25_reg_season.csv"
    ]

    # Initialize an empty list to hold DataFrames
    all_seasons_data = []

    # Loop through each file, load the data, and append to the list
    for season_file in season_files:
        file_path = os.path.join(data_dir, season_file)

        season_data = pd.read_csv(file_path)

        season_data["Opp"] = season_data["Opp"].replace({"PHO": "PHX", "CHO": "CHA"})

        # Drop rows where all columns are NaN
        season_data.dropna(how="all", inplace=True)

        # Append an extra column for the season/year, derived from the filename
        # Example: Extract "2021-22" from "curry_2021-22_reg_season.csv"
        season_year = season_file.split("_")[1]
        season_data["Season"] = season_year

        all_seasons_data.append(season_data)

    # Combine all seasons into a single DataFrame
    combined_data = pd.concat(all_seasons_data, ignore_index=True)

    return combined_data


def group_by_opponent(group_team_data):
    """
    Groups the combined DataFrame by the 'Opp' column and handles inactive games.
    - Converts all numeric columns to proper numeric dtype, excluding games where stats are marked as "Inac".
    - Returns grouped data for numeric calculations and a DataFrame for inactive games.
    """

    # List of columns to treat as numeric (list all stats columns)
    numeric_columns = [
        "FG", "FGA", "FG%", "3P", "3PA", "3P%", "2P", "2PA", "2P%",
        "eFG%", "FT", "FTA", "FT%", "ORB", "DRB", "TRB", "AST", "STL",
        "BLK", "TOV", "PF", "PTS", "GmSc", "+/-"
    ]

    # Separate inactive games (where Curry did not play)
    inactive_games = group_team_data[group_team_data["PTS"] == "Inac"]

    # Exclude inactive games and convert numeric columns
    numeric_data = group_team_data[group_team_data["PTS"] != "Inac"].copy()
    for col in numeric_columns:
        # Convert each column to numeric, coercing errors to NaN
        numeric_data[col] = pd.to_numeric(numeric_data[col], errors="coerce")

    # Group active (numeric) games by 'Opp'
    grouped_active = numeric_data.groupby("Opp")

    return grouped_active, inactive_games

def calculate_points_stats(grouped_active):
    """
    Calculates total and average points scored against each opponent.
    Ensures that all numeric precision is preformatted to remove trailing zeros.
    """

    points_stats = []

    # Iterate through each team in grouped_active
    for team, games in grouped_active:
        total_points = round(games["PTS"].sum(), 2)  # Enforce .00 precision
        average_points = round(games["PTS"].mean(), 2)


        points_stats.append({
            "Opponent": team,
            "Total Points": total_points,
            "Average Points": average_points
        })

    # Convert the results to a DataFrame
    points_summary = pd.DataFrame(points_stats)

    # Ensure columns are numeric (not strings)
    points_summary["Total Points"] = pd.to_numeric(points_summary["Total Points"], errors="coerce")
    points_summary["Average Points"] = pd.to_numeric(points_summary["Average Points"], errors="coerce")

    # Sort the DataFrame by Total Points in descending order
    points_summary = points_summary.sort_values(by="Total Points", ascending=False)

    return points_summary

# --- ORGANIZED DATA - PULL FROM HERE ---
def calculate_team_stats(grouped_active, stat_columns):
    """
    Calculates total, average, and count of games for given stats columns,
    grouped by the opponent, and ensures all values are preformatted
    to show exactly two decimal places without scientific notation.
    """

    team_stats = []

    # Iterate through each opponent in grouped_active
    for team, games in grouped_active:
        # Create a dictionary to store the team's stats
        stats = {"Opponent": team, "# Games": len(games)}

        # Calculate total and average for each stat
        for stat in stat_columns:
            total = round(games[stat].sum(), 2)
            average = round(games[stat].mean(), 2)
            stats[f"Total {stat}"] = total
            stats[f"Average {stat}"] = average

        # Append the dictionary to the list
        team_stats.append(stats)

    # Convert the collected stats into a DataFrame
    stats_summary = pd.DataFrame(team_stats)

    # Enforce formatting
    numeric_columns = stats_summary.select_dtypes(include="number").columns
    stats_summary[numeric_columns] = stats_summary[numeric_columns].apply(
        lambda col: col.map(
            lambda x: "{:.2f}".format(x) if pd.notnull(x) else x  # Exactly 2 decimal places
        )
    )

    # Sort by a relevant column
    if "Total PTS" in stats_summary.columns:
        stats_summary = stats_summary.sort_values(by="Total PTS", ascending=False)

    return stats_summary

def calculate_filter_stats(stats_summary):
    """
    Extracts and organizes stats from the provided stats summary and calculates combined metrics ( Avg PTS + AST, Avg PTS + REB)
    at the end of the columns.
    """

    filter_columns = ["Opponent", "Average PTS", "Average AST", "Average TRB",
                        "Average 3P", "Average STL", "Average BLK"]

    # Extract relevant columns
    filter_stats = stats_summary[filter_columns].copy()

    # Ensure numeric columns are properly converted
    numeric_columns = filter_stats.columns.difference(["Opponent"])
    for col in numeric_columns:
        filter_stats[col] = pd.to_numeric(filter_stats[col], errors="coerce")

    # Add the combined metrics at the end
    filter_stats["Avg PTS + AST"] = filter_stats["Average PTS"] + filter_stats["Average AST"]
    filter_stats["Avg PTS + REB"] = filter_stats["Average PTS"] + filter_stats["Average TRB"]

    # Reorder the columns to ensure combined metrics appear last
    final_columns = filter_columns + ["Avg PTS + AST", "Avg PTS + REB"]
    filter_stats = filter_stats[final_columns]

    # Round all numeric columns to 2 decimal places
    filter_stats = filter_stats.round(2)

    filter_stats.sort_values("Average PTS", ascending=False, inplace=True)

    return filter_stats


