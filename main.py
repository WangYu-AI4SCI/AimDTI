# -*- coding:utf-8 -*-

import argparse

from train import train, tower_train, cat_train

parser = argparse.ArgumentParser(prog='AimDTI')

parser.add_argument('-d', '--dataSetName', choices=[
                    "DrugBank", "KIBA", "Davis", "Enzyme", "GPCRs", "ion_channel"], default='KIBA', help='Enter which dataset to use for the experiment')
parser.add_argument('-m', '--model', choices=['Tower',  'GEDTI', 'AimDTI'],
                    default='AimDTI', help='Which model to use, \"AimDTI\" is used by default')
parser.add_argument('-s', '--seed', type=int, default=114514,
                    help='Set the random seed, the default is 114514')
parser.add_argument('-l', '--loss', default='FocalLoss', choices=['CELoss', 'PolyLoss', 'FocalLoss'],
                    help='Set the loss function, \"CELoss\" is used by default')
args = parser.parse_args()


if args.model == 'AimDTI':
    train(SEED=args.seed, DATASET=args.dataSetName, MODEL=args.model, LOSS=args.loss)
elif args.model == 'Tower':
    tower_train(SEED=args.seed, DATASET=args.dataSetName, MODEL=args.model, LOSS=args.loss)
elif args.model == 'GEDTI':
    cat_train(SEED=args.seed, DATASET=args.dataSetName, MODEL=args.model, LOSS=args.loss)