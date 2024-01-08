import logging

import click

from src.backend import generate_er_datasets, generate_er_graphs, train_er_model


# ---------------------------------------------------------------------------------------------#
# Generate Dataset #
# ---------------------------------------------------------------------------------------------#
@click.group()
def cli_gen_dataset():
    """
    Entrypoint for generating dataset.
    """
    pass


@click.command()
@click.option(
    "--directory",
    default="data/copy20221006/documents/train_19700101_20210401",
    help="Path to directory containing JSON files",
)
@click.option("--output_csv", default="data/test.csv", help="Path to output CSV file")
@click.option(
    "--output_cve_csv", default="data/cve_docs.csv", help="Path to output CSV file containing only documents with CVEs"
)
@click.option(
    "--output_cveless_csv",
    default="data/cveless_docs.csv",
    help="Path to output CSV file containing only documents without CVEs",
)
@click.option("--regen_data", type=bool, default=True, help="Whether to regenerate the CSV files. Defaults to True.")
@click.option(
    "--limit", type=int, default=None, help="Limit the number of JSON files to process. Defaults to None (no limit)."
)
def generate_dataset(directory, output_csv, output_cve_csv, output_cveless_csv, regen_data, limit):
    # Your main function logic goes here
    generate_er_datasets.main(
        directory=directory,
        output_csv=output_csv,
        output_cve_csv=output_cve_csv,
        output_cveless_csv=output_cveless_csv,
        regen_data=regen_data,
        limit=limit,
    )


# ---------------------------------------------------------------------------------------------#
# Generate Graph #
# ---------------------------------------------------------------------------------------------#
@click.group()
def cli_gen_graph():
    """
    Entrypoint for generating graph structure.
    """
    pass


@cli_gen_graph.command()
@click.option("--read_path", type=str, default="data/cve_docs.csv", help="Path to the CSV file containing CVE data.")
@click.option("--output_dir", type=str, default="data/", help="Path to the output directory.")
@click.option("--limit", type=int, default=None, help="Limit the number of rows to read from the CSV file.")
def gen_graph(read_path, output_dir, limit):
    """
    Main logic to generate graph structure.
    """
    generate_er_graphs.main(read_path, output_dir, limit)


# ---------------------------------------------------------------------------------------------#
# Train ER Model #
# ---------------------------------------------------------------------------------------------#
@click.group()
def cli_train():
    """
    Entrypoint for training ER model.
    """
    pass


@cli_train.command()
@click.option("--read_dir", default="data", help="Path to the directory containing the files to read.")
@click.option("--limit", default=102, type=int, help="Number of samples to use for training.")
@click.option("--num_layers", default=2, type=int, help="List of hidden dimensions for each layer.")
@click.option("--checkpoint_path", default="models/checkpoint.pth.tar", help="Path to save the model checkpoint.")
@click.option("--load_from_checkpoint", is_flag=True, help="Load best model checkpoint if available.")
@click.option("--train_percent", default=0.80, help="Percentage of data to use for training.")
@click.option("--valid_percent", default=0.20, help="Percentage of data to use for validation.")
@click.option("--num_epochs", default=100, help="Number of training epochs.")
@click.option("--learning_rate", default=0.01, help="Learning rate for the optimizer.")
@click.option("--weight_decay", default=1e-5, help="Weight decay for the optimizer.")
@click.option("--dropout_rate", default=0.0, help="Dropout rate for the model.")
@click.option("--plot_results", is_flag=True, default=True, help="Plot training and validation loss.")
@click.option("--logging_interval", default=50, help="Logging interval for metrics.")
def train(
    read_dir,
    limit,
    num_layers,
    checkpoint_path,
    load_from_checkpoint,
    train_percent,
    valid_percent,
    num_epochs,
    learning_rate,
    weight_decay,
    dropout_rate,
    plot_results,
    logging_interval,
):
    """
    Main logic to grab data, train model, and plot results.
    """
    # Additional arguments are passed as keyword arguments
    kwargs = {
        "train_percent": train_percent,
        "valid_percent": valid_percent,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "dropout_rate": dropout_rate,
        "plot_results": plot_results,
        "logging_interval": logging_interval,
        "load_from_checkpoint": load_from_checkpoint,
    }

    train_er_model.main(read_dir, limit, num_layers, checkpoint_path, **kwargs)


# ---------------------------------------------------------------------------------------------#
# END OF ENTRYPOINT COMMANDS #
# ---------------------------------------------------------------------------------------------#
cli = click.CommandCollection(sources=[cli_train, cli_gen_graph])

if __name__ == "__main__":
    logging.basicConfig(
        format="[ %(asctime)s.%(msecs)03d - %(levelname)s - %(filename)s:%(lineno)d ] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    cli()
