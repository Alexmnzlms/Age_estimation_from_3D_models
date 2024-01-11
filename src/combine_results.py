import argparse
import logging
import os
import pandas as pd
import numpy as np

def main(path, netname):
    result_dict = {}
    for root, folders, files in os.walk(path):
        for file in files:
            filename, ext = os.path.splitext(file)
            filepath = os.path.join(root,file)
            if ext == ".csv":
                if filename.startswith("execution_results"):
                    if filename.endswith(netname):
                        model = os.path.basename(os.path.dirname(filepath))
                        dataframe = pd.read_csv(filepath, sep=";")
                        print(model)
                        print(dataframe.to_string())
                        print()
                        results = dataframe.to_numpy()
                        for phase in results:
                            phase_name = phase[1]
                            if phase_name not in result_dict:
                                result_dict[phase_name] = [np.array(phase[2:])]
                            else:
                                result_dict[phase_name].append(np.array(phase[2:]))

    for key in result_dict.keys():
        result_dict[key] = np.mean(result_dict[key], axis=0)

    dataframe = pd.DataFrame.from_dict(result_dict).T

    dataframe = dataframe.rename(columns={0:"MSE", 1:"RMSE", 2:"MAE", 3:"R2"})

    print(dataframe.to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        "--folder",
        type=str,
        help="",
        required=True
    )

    parser.add_argument(
        "-n",
        "--name",
        type=str,
        help="",
        required=False,
        default="RESNET50"
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

    main(args.folder, args.name)