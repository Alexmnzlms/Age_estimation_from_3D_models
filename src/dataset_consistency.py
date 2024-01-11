import argparse
import logging
import pandas as pd
import numpy as np
import os

def main(data, csv, output):
    list_dir = [ name for name in os.listdir(data) if os.path.isdir(os.path.join(data, name)) ]

    dataframe = pd.read_csv(csv)

    filename_csv = os.path.basename(csv)

    sample_list = dataframe["Sample"].to_list()

    sample_list_new = []

    n_missing = 0
    for sample in sample_list:
        if sample not in list_dir:
            n_missing += 1
            print("Sample {} missing from {}".format(sample, csv))
            dataframe = dataframe.drop(dataframe[dataframe['Sample'] == sample].index)
        else:
            sample_list_new.append(sample)

    print("Before: {} | After: {}".format(len(sample_list), len(list_dir)))
    print("Missing: {} | Before-Missing : {}".format(n_missing, len(sample_list) - n_missing))

    print(dataframe)

    dataframe.to_csv(os.path.join(output, filename_csv), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=str,
        help="",
        required=True
    )

    parser.add_argument(
        "--csv",
        type=str,
        help="",
        required=True
    )

    parser.add_argument(
        "--output",
        type=str,
        help="",
        required=True
    )

    parser.add_argument(
        "-v", 
        "--verbose", 
        type=int, 
        required=False, 
        default=0
    )


    args = parser.parse_args()

    log_level = logging.WARNING
    if args.verbose == 0:
        log_level = logging.WARNING
    elif args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose == 2:
        log_level = logging.DEBUG
    else:
        logging.warning('Log level not recognised. Using WARNING as default')

    logging.getLogger().setLevel(log_level)

    logging.warning("Verbose level set to {}".format(logging.root.level))

    assert os.path.abspath(os.path.dirname(args.csv)) != os.path.abspath(args.output)

    os.makedirs(args.output, exist_ok=True)

    main(args.data, args.csv, args.output)
