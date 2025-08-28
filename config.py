# -*- coding:utf-8 -*-


class hyperparameter:
    def __init__(self, DATASET, MODEL):
        #
        self.drug_max_lengh = 100
        self.protein_max_lengh = 1000
        self.drug_vocab_size = 65
        self.protein_vocab_size = 26

        self.Learning_rate = 1e-4
        self.Epoch = 150
        self.Tower_ending = 80
        self.Batch_size = 16  ## 16
        self.Tower_patience = 10
        self.Fusion_patience = 10
        self.decay_interval = 10
        self.lr_decay = 0.5
        self.weight_decay = 1e-4
        self.embed_dim = 64
        self.protein_kernel = [4, 8, 12]
        self.drug_kernel = [4, 6, 8]
        self.conv = 40
        self.char_dim = 64
        self.dropout_rate = 0.5
        # focal loss
        # KIBA 0.7 1.2
        # Davis 0.6 1.5
        self.alpha = 0.6
        self.gamma = 1.2
        self.beta = 0.1

        # poly loss
        self.loss_epsilon = 1

        # dir
        self.dir_train = 'data/{}/random/train.txt'.format(DATASET)
        self.dir_valid = 'data/{}/random/valid.txt'.format(DATASET)
        self.dir_test = 'data/{}/random/test.txt'.format(DATASET)

        self.embedding_path = "association_learner/save_embedding/{}".format(DATASET)
        self.save_path = "./results/" + DATASET + "/" + MODEL
        # self.save_path = "./results/" + DATASET
