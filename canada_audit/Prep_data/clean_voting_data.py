import pandas as pd
import glob

# Retrieved from https://www.elections.ca/res/rep/off/ovr2019app/51/data_donnees/pollbypoll_bureauparbureauCanada.zip
filenames = glob.glob("../Raw_data/electoral/*.csv")

results_data = pd.DataFrame({})

for filename in filenames:
    poll_result = pd.read_csv(filename)

    electoral_district_number = poll_result.filter(
        like="Electoral District Number"
    ).iloc[0, 0]

    electoral_district_name = poll_result.filter(like="Electoral District Name").iloc[
        0, 0
    ]

    num_rejected_ballots = poll_result.filter(like="Rejected Ballots").iloc[0, 0]

    parties = (
        poll_result.filter(like="Political Affiliation Name_English")
        .iloc[:, 0]
        .unique()
    )

    vote_totals = poll_result.groupby(
        ["Political Affiliation Name_English/Appartenance politique_Anglais"]
    ).apply(
        lambda x: sum(x["Candidate Poll Votes Count/Votes du candidat pour le bureau"])
    )

    results_row = pd.DataFrame(
        {
            "District number": [electoral_district_number],
            "District name": [electoral_district_name],
            "Invalid ballots": [num_rejected_ballots],
        }
    )

    results_row = pd.concat(
        [results_row, pd.DataFrame(vote_totals).transpose()], axis=1
    )

    results_data = pd.concat([results_data, results_row], axis=0)

# Now that the data are in a reasonable format, let's combine some columns
main_parties = [
    "Liberal",
    "Conservative",
    "Green Party",
    "NDP-New Democratic Party",
    "People's Party",
    "Bloc Québécois",
    "Independent",
]
results_data = results_data.fillna(0)

other_votes = results_data.drop(main_parties, axis=1).apply(
    lambda row: row["Invalid ballots":"National Citizens Alliance"].sum(), axis=1
)

results_data["Other votes"] = other_votes

results_data = results_data[
    ["District number", "District name"] + main_parties + ["Other votes"]
]
results_data[main_parties + ["Other votes"]] = results_data[
    main_parties + ["Other votes"]
].astype("int64")

shorthand_mapping = {
    "Conservative": "PC",
    "NDP-New Democratic Party": "NDP",
    "People's Party": "PPC",
    "Bloc Québécois": "Bloc",
    "Green Party": "Green",
    "Independent": "Independent",
}

results_data["Winner_shorthand"] = (
    results_data[main_parties].idxmax(axis=1).replace(shorthand_mapping)
)

results_data["Total votes"] = results_data.loc[:, "Liberal":"Other votes"].sum(axis=1)

results_data.to_csv("../Data/Canada_2021_federal_election_results.csv", index=False)
