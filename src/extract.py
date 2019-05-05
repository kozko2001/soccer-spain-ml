import pandas as pd
import numpy as np

def get_grouped_events():
    events_df = pd.read_csv('../data/events.csv')
    return pd.pivot_table(events_df,
                          values='minuto',
                          index=['id_partido', 'id_equipo'],
                          columns=['evento'],
                          aggfunc='count',
                          fill_value=0)

def get_training():
    training = pd.read_csv('../data/train_matches.csv')
    ## droping players ids
    return training[['id_partido', 'id_equipo_local', 'id_equipo_visitante', 'ganador']]

def get_test():
    test = pd.read_csv('../data/test_matches.csv')
    return test[['id_partido', 'id_equipo_local', 'id_equipo_visitante']]


def join_matches_with_events(matches, grouped_events):
    no_events = matches
    only_local_events = no_events.merge(grouped_events, how='left', left_on=['id_partido', 'id_equipo_local'], right_on=['id_partido', 'id_equipo'])
    matches_with_events = only_local_events.merge(grouped_events, suffixes=("_local", "_visitante"),how='left', left_on=['id_partido', 'id_equipo_visitante'], right_on=['id_partido', 'id_equipo'])
    matches_with_events = matches_with_events.fillna(0)

    return matches_with_events

def make_label_last_colums(df):
    cols = list(df.columns)
    cols = list(filter(lambda name: name != "ganador", cols)) + ['ganador']
    df = df[cols]
    return df




def main():
    grouped_events = get_grouped_events()

    ## Get training data
    training_matches = get_training()
    training_data = join_matches_with_events(training_matches, grouped_events)
    training_data = make_label_last_colums(training_data)
    training_data.to_csv('../data/training_join.csv', index=False)

    ## Get test data
    test_matches = get_test()
    test_data = join_matches_with_events(test_matches, grouped_events)
    test_data.to_csv('../data/test_join.csv', index=False)

if __name__ == "__main__":
    main()
