import pandas as pd
matches = pd.read_csv("matches.csv",index_col = 0)
matches["team"].value_counts()
matches[matches["team"] == "Barcelona"]
matches["round"].value_counts()

#need to convert all data-type to numeric either float or int since ml can only work with that data

matches["date"] = pd.to_datetime(matches["date"])
#converts date to the pandas datetime datatype

matches["venue_code"] = matches["venue"].astype("category").cat.codes
#venue_code stores either a 0 to signify the team is away and a 1 to show the team is home

matches["opp_code"] = matches["opponent"].astype("category").cat.codes
#opponent_code stores a numeric value based on the opponent

matches["hour"] = matches["time"].str.replace(":.+", "", regex = True).astype("int")
#gets a time column that is only the hour 

matches["day_code"] = matches["date"].dt.dayofweek
#gets the day of the week as a number

matches["target"] = (matches["result"] == "W").astype("int")
#(matches["result"] == "W") returns a boolean whether it is a W
#then it gets stored as an int, 1 for true, 0 for false

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 101, min_samples_split = 150, random_state = 1) 
train = matches[matches["date"] < '2023-01-01']
test = matches[matches["date"] >= '2023-01-01']
predictors = ["venue_code", "opp_code",  "hour", "day_code"]
rf.fit(train[predictors], train["target"])
preds = rf.predict(test[predictors])

from sklearn.metrics import accuracy_score
acc = accuracy_score(test["target"], preds)
combined = pd.DataFrame(dict(actual = test["target"], prediction = preds), index = test.index)
pd.crosstab(index = combined["actual"], columns = combined["prediction"])

#0 represents loss or draw, therefore, 2165 we were correct, 1141 we were wrong
#1 represents wins so 608 we were correct, 546 we were wrong

from sklearn.metrics import precision_score

precision_score(test["target"], preds)
#when we predicted win, team only won 47% of time

grouped_matches = matches.groupby("team")
group = grouped_matches.get_group("Barcelona")

def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed = 'left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset = new_cols)
    #^deals with missing vals
    return group

cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]
rolling_averages(group, cols, new_cols)
matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling = matches_rolling.droplevel('team')
matches_rolling.index = range(matches_rolling.shape[0])

def make_predictions(data, predictors):
    train = data[data["date"] < '2023-01-01']
    test = data[data["date"] >= '2023-01-01']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual = test["target"], prediction = preds), index = test.index)
    precision = precision_score(test["target"], preds)
    return combined, precision

combined, precision = make_predictions(matches_rolling, predictors + new_cols)
combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index = True, right_index = True)

class MissingDict(dict):
    __missing__  = lambda self, key : key
map_values = {
}
mapping = MissingDict(**map_values)


combined["new_team"] = combined["team"].map(map_values)
merged = combined.merge(combined, left_on = ["date", "new_team"], right_on = ["date", "opponent"])
print(merged[(merged["prediction_x"] == 1) & (merged["prediction_y"] == 0)]["actual_x"].value_counts())




