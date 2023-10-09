# -*- coding: utf-8 -*-
import click
from gen_features import main as gen_features_main
from score_data import main as score_data_main
from train_model import main as train_model_main


# This is assuming you have these functions in a script named "your_script_name.py". Replace with your actual script/module name.
@click.group()
def cli():
    """Your Main CLI"""
    pass


@click.command()
@click.option(
    "--exploit_data_directory",
    default="data/copy20221006/files/train_19700101_20210401/exploits_text",
    show_default=True,
    help="Path to the directory containing exploit JSON files.",
)
@click.option(
    "--documents_path",
    default="data/copy20221006/files/train_19700101_20210401/documents.json",
    show_default=True,
    help="Path to the documents JSON file.",
)
@click.option(
    "--labels_directory",
    default="data/copy20221006/documents/train_19700101_20210401",
    show_default=True,
    help="Path to the directory containing label JSON files.",
)
@click.option(
    "--export_path",
    default="data/data",
    required=True,
    help="Path where the merged CSV will be saved (without .csv extension).",
)
def generate_features(exploit_data_directory, documents_path, labels_directory, export_path):
    """Main CLI function to process exploit data and labels and save merged results to a CSV."""
    gen_features_main(exploit_data_directory, documents_path, labels_directory, export_path)


@click.command()
@click.option("--data_path", default="data/data.csv", show_default=True, help="Path to the data")
@click.option("--export_model_path", default="models/model", show_default=True, help="Model path")
def train_model(data_path, export_model_path):
    """Main CLI function to process exploit data and labels and save merged results to a CSV."""
    train_model_main(data_path, export_model_path)


@click.command()
@click.option("--input_data_path", default="data/data.csv", show_default=True, help="Path to the data")
@click.option("--model_path", default="models/model", show_default=True, help="Model path")
@click.option("--vectorizer_path", default="models/model_vectorizer", show_default=True, help="vec path")
@click.option("--out_path", default="data/scored_data", show_default=True, help="scored data path")
def score_data(input_data_path, model_path, vectorizer_path, out_path):
    """Main CLI function to process exploit data and labels and save merged results to a CSV."""
    score_data_main(input_data_path, model_path, vectorizer_path, out_path)


cli.add_command(generate_features)
cli.add_command(train_model)
cli.add_command(score_data)

if __name__ == "__main__":
    cli()
