# Author: Konstantinos Dimitriou
# Date: 10/6/2026

import numpy as np
import pandas as pd
import os

def logistic_solution(t, u0, r):
    return (u0 * np.exp(r*t)) / ((1-u0) + u0 * np.exp(r*t))


def generate_data_dictionary(u0_values, r_values, t_values):
    # Build base dictionary
    data = {"u0" : [], "r" : []}
    for i in range(len(t_values)):
        data[f"t{i}"] = []
    # Fill dictionary
    for u0 in u0_values:
        for r in r_values:
            solution = logistic_solution(t_values, u0, r)
            data["u0"].append(u0)
            data["r"].append(r)
            for i in range(len(t_values)):
                data[f"t{i}"].append(solution[i])
    return data


def main():
    # Set random seed for reproducibility
    np.random.seed(17)

    # Set time grid
    t_values = np.linspace(0,1,100)

    # Generate training and testing constants (100*100=10000, 32*32 = 1024)
    u0_all = np.random.uniform(0.01, 0.99, 132)
    u0_train, u0_test = u0_all[:100], u0_all[100:]
    r_all = np.random.uniform(0.1, 10.0, 132)
    r_train, r_test = r_all[:100], r_all[100:]

    # Generate solutions and build data
    dataframe_training = pd.DataFrame(data=generate_data_dictionary(u0_train, r_train, t_values))
    dataframe_testing = pd.DataFrame(data=generate_data_dictionary(u0_test, r_test, t_values))
    # Save data to csv file
    os.makedirs("DATA", exist_ok=True)
    dataframe_training.to_csv("DATA/training_data.csv", index=False)
    dataframe_testing.to_csv("DATA/testing_data.csv", index=False)


if __name__=="__main__":
    main()

