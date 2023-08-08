import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # Training parameters
    parser.add_argument('--num_classes', type=int, default=6, help="number of classes")
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    parser.add_argument('--data_path', default="./TrainingSet/")
    parser.add_argument('--save_path', default='./saved_model/')
    parser.add_argument('--spd_para', type=float, default=0.2)
    parser.add_argument('--type', default='all')
    parser.add_argument('--version', type=int, default=1, help="cross validation version (1-5)")
    parser.add_argument('--batch_train', type=int, default=32, help="batch size of training set")
    parser.add_argument('--batch_test', type=int, default=32, help="batch size of testing set")
    parser.add_argument('--seed', type=int, default=123, help="random seed")
    parser.add_argument('--sample_size0', type=int, default=0, help="sample size for each class")
    parser.add_argument('--sample_size1', type=int, default=0, help="sample size for each class")
    parser.add_argument('--sample_size2', type=int, default=0, help="sample size for each class")
    parser.add_argument('--sample_size3', type=int, default=0, help="sample size for each class")
    parser.add_argument('--sample_size4', type=int, default=0, help="sample size for each class")
    parser.add_argument('--sample_size5', type=int, default=0, help="sample size for each class")
    parser.add_argument('--std', type=float, default=0.01)
    parser.add_argument('--new_data', default='false')

    # Testing parameters
    parser.add_argument('--final', default='false')
    parser.add_argument('--testing_data_path', default="./TestingSet/")
    parser.add_argument('--batch', type=int, default=32, help="final batch size of testing set")



    args, _ = parser.parse_known_args()
    return args
