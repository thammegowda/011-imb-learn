#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 7/19/21
import copy
from pathlib import Path

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import create_scheduler
import torch

from imblearn import yaml, registry, MODEL, device, log
from imblearn.common.tune import find_tunable_params, update_conf_values

def objective(step, alpha, beta):
    return (0.1 + alpha * step / 100)**(-1) + beta * 0.1


def training_function(config):
    # Hyperparameters
    print("Config>>>", config)
    alpha, beta = config["alpha"], config["beta"]
    for step in range(10):
        # Iterative training function - can be any arbitrary training procedure.
        intermediate_score = objective(step, alpha, beta)
        # Feed the score back back to Tune.
        tune.report(mean_loss=intermediate_score)



def _train(config, pipeline=None, checkpoint_dir=None):
    #assert pipeline
    restore_checkpoint = None
    if checkpoint_dir:
        restore_checkpoint = Path(checkpoint_dir) / 'checkpoint'
    conf = update_conf_values(copy.deepcopy(pipeline.conf), named_values=config,
                              name_to_path=pipeline.name_to_path.get)
    trial_dir = Path(tune.get_trial_dir())
    yaml.dump(conf, trial_dir / 'conf.yml')
    # trainer = pipeline.model_cls.Trainer(trial_dir, restore_checkpoint=restore_checkpoint,
    #                                 device=device)
    # trainer.train_()


def dummy(config,):
    print(config)
    for i in range(10):
        tune.report(loss=torch.rand())


class Pipeline:

    def __init__(self, exp_dir: Path):
        self.exp_dir = exp_dir
        conf_file = exp_dir / 'conf.yml'
        conf_tune_file = exp_dir / 'conf.tune.yml'
        assert conf_file.exists() or conf_tune_file.exists(), \
            f'Either conf.yml or conf.tune.yml is expected in {exp_dir}, but it none found.'
        if conf_file.exists():
            # either already tuned or tuning is not required
            self.tuning = False
            self.conf = yaml.load(conf_file)
        else:
            self.tuning = True
            assert conf_tune_file.exists()
            self.conf = yaml.load(conf_tune_file)
            tunables = list(find_tunable_params(self.conf))
            assert tunables, 'No tunable params found'
            log.info(f"Found {len(tunables)} tunable params")
            # assumption : path is list of segments; where each item is key in dict or index in list
            tunables = {tuple(path): node for path, node in tunables}
            self.name_to_path = {path[-1]: path for path in tunables.keys()}
            assert len(self.name_to_path) == len(tunables)  # no conflicts with name
            # get last portion of path as name --> TunableParam.param
            self.param_space = {path[-1]: node.param for path, node in tunables.items()}
            assert len(self.param_space) == len(tunables)  # no conflicts with name

        assert MODEL in self.conf and 'name' in self.conf['model']
        self.model_name = self.conf[MODEL]['name']
        assert self.model_name in registry[MODEL]
        self.model_cls = registry[MODEL][self.model_name]

    def run_tuner(self):
        assert self.tuning
        tune_conf = self.conf['tune']
        sched_name, sched_args = tune_conf['scheduler']['name'], tune_conf['scheduler']['args']
        log.info(f"Creating tuning scheduler: {sched_name}   args: {sched_args}")
        scheduler = create_scheduler(sched_name, **sched_args)
        reporter = CLIReporter(
            # parameter_columns=["l1", "l2", "lr", "batch_size"],
            metric_columns=["loss", "accuracy", "training_iteration"])
        run_args = tune_conf['run']
        log.info(f"Tuner run args: {run_args}")
        config = {
            "alpha": tune.grid_search([0.001, 0.01, 0.1]),
            "beta": tune.choice([1, 2, 3])
        }
        result = tune.run(
            #partial(_train, pipeline=self),
            training_function,
            local_dir=str(self.exp_dir),
            #config=self.param_space,
            config=config,
            scheduler=scheduler,
            progress_reporter=reporter,
            #resume=True,
            **run_args)

        best_trial = result.get_best_trial(sched_args['metric'], sched_args['mode'], "last")
        if not best_trial:
            log.warning("No best trial found. Exiting.")
            return
        print(f"Best trial config: {best_trial.config}")
        print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
        best_conf_yml = update_conf_values(copy.deepcopy(self.conf), named_values=best_trial.config,
                                           name_to_path=self.name_to_path.get)
        best_conf_yml_path = self.exp_dir / 'conf.yml'
        log.info(f'Storing best config at {best_conf_yml_path}')
        yaml.dump(best_conf_yml, best_conf_yml_path)
        self.tuning = False  # done

        """
        print("Best trial final validation accuracy: {}".format(
            best_trial.last_result["accuracy"]))
        best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if gpus_per_trial > 1:
                best_trained_model = nn.DataParallel(best_trained_model)
        best_trained_model.to(device)
    
    
        best_checkpoint_dir = best_trial.checkpoint.value
        model_state, optimizer_state = torch.load(Path(best_checkpoint_dir) / "checkpoint")
        best_trained_model.load_state_dict(model_state)
    
        #test_acc = test_accuracy(best_trained_model, device)
        #print("Best trial test set accuracy: {}".format(test_acc))
        """

    def run(self):
        if self.tuning:
            self.run_tuner()

        assert not self.tuning  # tuning finished
        trainer = self.model_cls.Trainer(self.exp_dir, device=device)
        trainer.pipeline()


def parse_args():
    import argparse
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('exp_dir', type=Path, help='Experiment dir path (must have conf.yml in it).')
    return p.parse_args()


def main(**args):
    args = args or vars(parse_args())
    exp_dir: Path = args['exp_dir']
    pipe = Pipeline(exp_dir)
    pipe.run()


if __name__ == '__main__':
    main()
