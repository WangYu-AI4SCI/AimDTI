import datetime

from data.data import Data
from util.logger import Log
from os.path import abspath
from time import strftime, localtime, time


class Recommender:
    def __init__(self, conf, training_set, validation_set, test_set, **kwargs):
        self.config = conf
        self.data = Data(self.config, training_set, validation_set, test_set)

        # Store references in class attributes for repeated use
        model_config = self.config['model']
        self.model_name = model_config['name']
        self.ranking = self.config['item.ranking.topN']
        self.emb_size = int(self.config['embedding.size'])
        self.maxEpoch = int(self.config['max.epoch'])
        self.batch_size = int(self.config['batch.size'])
        self.lRate = float(self.config['learning.rate'])
        self.reg = float(self.config['reg.lambda'])
        self.save_dir = self.config['save_dir']
        self.save_embedding = self.config['save_embedding']
        self.DataName = self.config['DataName']
        self.patience = self.config['patience']
        # Use f-string for better readability
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # self.model_log = Log(self.model_name, f"{self.model_name} {self.timestamp}")

        self.params = {
            "model_name": self.model_name,
            "data_name": self.DataName,
            "max_epoch": self.maxEpoch,
            "embedding_size": self.emb_size,
            "learning_rate": self.lRate,
            "regularization": self.reg,
            "patience": self.patience,
            "Training Set": abspath(self.config['training.set']),
            "Test Set": abspath(self.config['test.set'])
        }
    # def initializing_log(self):
    #     self.model_log.add('### model configuration ###')
    #     config_items = self.config.config
    #     for k in config_items:
    #         self.model_log.add(f"{k}={str(config_items[k])}")

    def print_model_info(self):
        for key, value in self.params.items():
            print(f"{key}: {value}")

        model_name = self.config['model']['name']
        if self.config.contain(model_name):
            args = self.config[model_name]
            parStr = '  '.join(f"{key}:{args[key]}" for key in args)
            print('Specific parameters:', parStr)

    def build(self):
        pass

    def train(self):
        pass

    def predict(self, u):
        pass

    def test(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def evaluate(self):
        pass


    def execute(self):
        # self.initializing_log()
        self.print_model_info()
        print('Initializing and building model...')
        self.build()
        print('Training Model...')
        self.train()
        print('Final Evaluating...')
        results_train, results_valid, results_test = self.evaluate()

        with open(self.save_dir + "/The_results_of_whole_dataset.txt", 'a') as f:
            f.write("Final results of ")
            f.write(f"{self.timestamp}\n")
            for k, v in self.params.items():
                if k == 'model_name' or k == 'data_name':
                    f.write(f"{k}: {v}\t")
                if k == 'max_epoch' or k == 'embedding_size' or k == 'regularization':
                    f.write(f"\n{k}: {v}")
            f.write('\n' + results_train)
            f.write('\n' + results_valid)
            f.write('\n' + results_test+ '\n\n')

        with open(self.save_embedding + "/readme.md.txt", "a") as f:
            f.write("Embedding note.md: ")
            f.write(f"{self.timestamp}\n")
            # f.write(f"{MODEL}\t{DATASET}\t{LOSS}\n")
            for k, v in self.params.items():
                f.write(f"{k}: {v}\n")
