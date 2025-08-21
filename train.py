from datetime import datetime
from pathlib import Path
import configargparse
import yaml
from modules.utils.configargparse_arguments import build_configargparser
from modules.utils.utils import (
    get_class_by_path,
)
from modules.utils.seed import seed

def train(hparams, ModuleClass, ModelClass, DatasetClass):
    """Main training routine

    Args:
        hparams: parameters for training
        ModuleClass (LightningModule): Contains training routine, etc.
        ModelClass: Contains network to train
        DatasetClass: Contains dataset on which to train
    """

    # load model
    model = ModelClass(hparams=hparams)
    # cross validation
    yaml_file = 'cross_val.yml'
    validation_splits = load_validation_splits_from_yaml(yaml_file)
    # for idx in range(0, hparams.num_patients):
    for idx in range(0, len(validation_splits)):

        # load dataset
        dataset = DatasetClass(hparams=hparams, idx=idx, val_splits=validation_splits)
        # load module
        module = ModuleClass(hparams, model, dataset, idx=idx)

        seed(0, hparams)
        module.train()

def load_validation_splits_from_yaml(yaml_file):
        """define validation splits"""
        with open(yaml_file, 'r') as f:
            splits_data = yaml.safe_load(f)
        return [split['validation'] for split in splits_data['splits']]

def main():
    """Main"""
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('-c', is_config_file=True, help='config file path')
    parser, hparams = build_configargparser(parser)

    # LOAD MODULE
    module_path = f"modules.{hparams.module}"
    ModuleClass = get_class_by_path(module_path)
    parser = ModuleClass.add_module_specific_args(parser)

    # LOAD MODEL
    model_path = f"models.{hparams.model}"
    ModelClass = get_class_by_path(model_path)
    parser = ModelClass.add_model_specific_args(parser)

    # LOAD DATASET
    dataset_path = f"datasets.{hparams.dataset}"
    DatasetClass = get_class_by_path(dataset_path)
    parser = DatasetClass.add_dataset_specific_args(parser)

    # PRINT PARAMS & INIT LOGGER
    hparams = parser.parse_args()

    date_str = datetime.now().strftime("%y%m%d-%H%M%S")

    hparams.name = hparams.subproject_name + '_' + date_str #+ exp_name
    hparams.output_path = Path(hparams.output_path).absolute() / hparams.name

    train(hparams, ModuleClass, ModelClass, DatasetClass)

if __name__ == "__main__":
    main()
