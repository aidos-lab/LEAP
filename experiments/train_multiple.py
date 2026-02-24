import click
from experiments.train_gcn import training_pipeline
from experiments.experiments_list import EXPERIMENTS_LIST as EXPERIMENTS
from utils.utility import dict2csv
from utils.config import Config

@click.command()
@click.option("--dataset", default="Letter-high", help="Dataset name")
@click.option("--run-id", default="", help="id to be appended to the output file name")
def main(dataset, run_id):
    dataset_name = dataset
    filename =  dataset_name + '_' +run_id + '_results.csv'
    for idx, experiment in enumerate(EXPERIMENTS):
        print("========================================================")
        print(f"Running experiment {idx + 1}: {experiment['description']}")
        config = Config(**Config.filter_dict(experiment))
        config.dataset_name = dataset_name
        for i in range(5):
            config.fold = i
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
            result = training_pipeline(**config.to_dict())
            dict2csv(result, filename)
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
        
        with open(filename, 'a') as f:
            f.write('\n')


if __name__ == "__main__":
    main()