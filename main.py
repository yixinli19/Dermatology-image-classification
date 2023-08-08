from torch.utils.data import DataLoader
from data import MyDataset
import os
import torch
import torch.nn as nn
import torchvision.models as models
from argparser import args_parser
from experiment_log import  PytorchExperimentLogger

import random
import numpy as np
import datetime
import time
import ast


torch.cuda.empty_cache()


#calculate the number of correct predictions
def num_correct_pred(output, target, class_num=None):
    if class_num != None:
        mask = (target == class_num)
        output = output[mask]
        target = target[mask]

    _, pred = torch.max(output, dim=1)
    correct_predictions = torch.sum(pred == target).item()

    return correct_predictions


def add_white_gaussian_noise(data, mean=0.0, std_dev=1.0):
    noise = np.random.normal(loc=mean, scale=std_dev, size=data.shape)
    result = data + noise
    return result.to(data.dtype)


# Define the loss function (knowledge distillation)
class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, class_weights, alpha=0.5, temperature=4.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.cross_entropy = torch.nn.CrossEntropyLoss(weight = class_weights).cuda()
        self.kl_divergence = torch.nn.KLDivLoss(reduction='batchmean')

    def forward(self, outputs, targets, teacher_outputs):
        ce_loss = self.cross_entropy(outputs, targets)
        kd_loss = self.kl_divergence(
            torch.log_softmax(outputs / self.temperature, dim=1),
            torch.softmax(teacher_outputs / self.temperature, dim=1)
        )
        loss = (1 - self.alpha) * ce_loss + self.alpha * self.temperature**2 * kd_loss
        return loss


#training
def train(train_loader, model, teacher_model, criterion, optimizer, args, confusion_matrix):
    model.train()
    if teacher_model != None:
        teacher_model.eval()

    running_loss = 0.0

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=4, active=3, repeat=1),
        # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/model'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for i, (images, targets, G, iteration) in enumerate(train_loader):
            
            for idx in range(len(iteration)):
                if iteration[idx] == 1:
                    images[idx] = add_white_gaussian_noise(images[idx], mean=0.0, std_dev=args.std)

            images = images.to(args.device, non_blocking=True)
            targets = targets.to(args.device, non_blocking=True)
            # print(images)

            output = model(images)
            _, pred = torch.max(output, dim=1)

            if teacher_model != None:
                teacher_outputs = teacher_model(images)
                loss = criterion(output, targets, teacher_outputs)
            else:
                loss = criterion(output, targets)
            running_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prof.step()

            # if torch.all(pred < 6):
            #     # for t, p in zip(targets.view(-1), pred.view(-1)):
            #     #     confusion_matrix[t.long(), p.long()] += 1
            #     for t, p, g in zip(targets.view(-1), pred.view(-1), G):
            #         g_val = 0
            #         if g == "G6":
            #             g_val = 0
            #         if g == "G7":
            #             g_val = 1
            #         if g == "G8":
            #             g_val = 2
            #         if g == "G10":
            #             g_val = 3
            #         confusion_matrix[g_val, t.int(), p.int()] += 1


    return running_loss



#testing
def test(test_loader, model, args, confusion_matrix):
    model.eval()
    
    for i, (images, targets, G, iteration) in enumerate(test_loader):
        # print(i)
        images = images.to(args.device)
        targets = targets.to(args.device)
        output = model(images)
        _, pred = torch.max(output, dim=1)

        if torch.all(pred < 6):
            for t, p, g in zip(targets.view(-1), pred.view(-1), G):
                g_val = 0
                if g == "G6":
                    g_val = 0
                if g == "G7":
                    g_val = 1
                if g == "G8":
                    g_val = 2
                if g == "G10":
                    g_val = 3
                confusion_matrix[g_val, t.int(), p.int()] += 1


    
    

def evaluate(test_sets_all, model, args):
    exp_logger.print("=" * 50 + '\n')
    exp_logger.print("Evaluating")
    exp_logger.print("="*50+'\n')
    # INIT LOGGERS
    
    NUM_ITERATIONS = 10
    
    NUM_IMAGES = 100
    L_MIN = 0.0
    L_MAX = 100.0

    latency_values = []
    
    for it in range(NUM_ITERATIONS):
        runtime_set = []

        while len(runtime_set) < NUM_IMAGES:
            for i, test_loader in enumerate(test_sets_all):
                for i, (images, targets, G, iteration) in enumerate(test_loader):
                    if random.random() < 0.5:
                        runtime_set.append(images)
                        if len(runtime_set) >= NUM_IMAGES:
                            break
                if len(runtime_set) >= NUM_IMAGES:
                    break
            

        print("size: ", len(runtime_set))

        # MEASURE PERFORMANCE
        # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
        # starter.record()
        start_time = time.time()
        with torch.no_grad():
            for img in runtime_set:
                img = img.to(args.device)
                _ = model(img)

        # ender.record()
        latency = time.time() - start_time
        # WAIT FOR GPU SYNC
        # torch.cuda.synchronize()
        # latency = starter.elapsed_time(ender) * 0.001

        latency_i = 1 - (latency - L_MIN) / (L_MAX - L_MIN)

        latency_values.append(latency_i)

    avg_latency = sum(latency_values) / len(latency_values)
    exp_logger.print(f"Inference Latency: {avg_latency}")
    exp_logger.print(f"Inference Latency List: {latency_values}")

    return avg_latency

    



def make_dataset(path, version, args, group, final_build):
    
    class_dict = {'BCC': 0, 'BKL': 1, 'MEL': 2, 'NV': 3, 'unknown': 4, 'VASC': 5} #label dictionary
    samples1 = []
    samples2 = []
    samples3 = []
    samples4 = []
    samples5 = []

    test_samples = []
    train_samples = []
    synthetic_train = []

    total_cnt = []
    init_cnt = []
    
    for class_name in class_dict:
        class_dir = os.path.join(path, class_name)
        label = class_dict[class_name]

        cnt = 0
        for file_name in os.listdir(class_dir):
            if file_name.endswith('.jpg') or file_name.endswith('.png'):
                img_path = os.path.join(class_dir, file_name)
                if random.random() < 0.2 :
                    samples1.append((img_path, label, 0))
                elif random.random() < 0.4 :
                    samples2.append((img_path, label, 0))
                elif random.random() < 0.6 :
                    samples3.append((img_path, label, 0))
                elif random.random() < 0.8 :
                    samples4.append((img_path, label, 0))
                elif random.random() < 1.0 :
                    samples5.append((img_path, label, 0))
                cnt += 1
        init_cnt.append(cnt)
        
        if version == 1:
            test_samples = samples1
            train_samples = samples2 + samples3 + samples4 + samples5

        if version == 2:
            test_samples = samples2
            train_samples = samples1 + samples3 + samples4 + samples5

        if version == 3:
            test_samples = samples3
            train_samples = samples2 + samples1 + samples4 + samples5

        if version == 4:
            test_samples = samples4
            train_samples = samples2 + samples3 + samples1 + samples5

        if version == 5:
            test_samples = samples5
            train_samples = samples2 + samples3 + samples4 + samples1

        
        sample_size = 0
        if label == 0:
            sample_size = args.sample_size0
        if label == 1:
            sample_size = args.sample_size1
        if label == 2:
            sample_size = args.sample_size2
        if label == 3:
            sample_size = args.sample_size3
        if label == 4:
            sample_size = args.sample_size4
        if label == 5:
            sample_size = args.sample_size5

        
        while cnt < sample_size and sample_size != 0:
            for i, l, j in train_samples:
                if random.random() < 0.8 :
                    synthetic_train.append((i, l, 1))
                    cnt += 1
                    if cnt >= sample_size:
                        break
                    
        total_cnt.append(cnt)
        
    train_samples += synthetic_train

    if final_build == "true":
        train_samples += test_samples
        return train_samples, test_samples, total_cnt

    exp_logger.print(f"train size final: {len(train_samples)}")
    exp_logger.print(f"test size final: {len(test_samples)}")

    exp_logger.print(f"total size: {total_cnt}")
    exp_logger.print(f"initial size: {init_cnt}")

    with open(f'log/test_samples_{group}.txt', 'w') as file:
        for item in test_samples:
            file.write(f'{str(item)}\n')

    with open(f'log/train_samples_{group}.txt', 'w') as file:
        for item in train_samples:
            file.write(f'{str(item)}\n')

    
    
    return train_samples, test_samples, total_cnt
    

def define_normalization(train_sets, test_sets):
    nimages = 0
    mean = 0.0
    var = 0.0

    for i, train_loader in enumerate(train_sets):
        for i_batch, batch_target in enumerate(train_loader):
            batch = batch_target[0]
            # Rearrange batch to be the shape of [B, C, W * H]
            batch = batch.view(batch.size(0), batch.size(1), -1)
            # Update total number of images
            nimages += batch.size(0)
            # Compute mean and std here
            mean += batch.mean(2).sum(0) 
            var += batch.var(2).sum(0)

            print(mean, var)

    for i, test_loader in enumerate(test_sets):
        for i_batch, batch_target in enumerate(test_loader):
            batch = batch_target[0]
            # Rearrange batch to be the shape of [B, C, W * H]
            batch = batch.view(batch.size(0), batch.size(1), -1)
            # Update total number of images
            nimages += batch.size(0)
            # Compute mean and std here
            mean += batch.mean(2).sum(0) 
            var += batch.var(2).sum(0)

            print(mean, var)

    mean /= nimages
    var /= nimages
    std = torch.sqrt(var)

    print(mean)
    print(std)


def performance_output(confusion_matrix, latency):
        
        exp_logger.print("Matrix: ")
        exp_logger.print(str(confusion_matrix))

        groups_acc = []

        for i in range(4):
            correct_cnt = 0
            all_cnt = 0
            for j in range(6):
                for k in range(6):
                    if j == k:
                        correct_cnt += confusion_matrix[i, j, k]
                    all_cnt += confusion_matrix[i, j, k]
            acc = float(correct_cnt / all_cnt)
            groups_acc.append(acc)

        overall_acc = sum(groups_acc) / len(groups_acc)
        exp_logger.print(f"Accuracy for each group (6-10): {groups_acc}")
        exp_logger.print(f"Overall Accuracy: {overall_acc}")
        
        abs_difference = 0
        minority_acc = groups_acc[3]
        for i in range(3):
            abs_difference += abs(groups_acc[i] - minority_acc)
        SPD = abs_difference / len(groups_acc)
        fairness_score = ((args.spd_para - SPD) / args.spd_para) / 3
        acc_score = overall_acc / 3
        latency_score = latency / 3
        exp_logger.print(f"SPD: {SPD}\n")
        exp_logger.print(f"Fairness score: {fairness_score}\n")
        exp_logger.print(f"Accuracy score: {acc_score}\n")
        exp_logger.print(f"Latency score: {latency_score}\n")
        exp_logger.print(f"Performance score: {acc_score + fairness_score + latency_score}\n")


if __name__ == '__main__':
 

    ## init setup
    exp_logger = PytorchExperimentLogger('./log', "log", ShowTerminal=True)
    args = args_parser()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(args.device)

    if args.final == "true":
        path = args.testing_data_path

        groups = os.listdir(path)
        if '.DS_Store' in groups: #remove other files
            groups.remove('.DS_Store')
        print(groups)

        data_sets = []
        
        #put DataLoader with different groups
        for i in range(len(groups)):
            print(path + groups[i])
            
            if args.new_data == "true":
                train_samples, test_samples, total_cnt = make_dataset(path + groups[i], 1, args, groups[i], final_build=args.final)
                data = MyDataset(groups[i], "test", train_samples, test_samples)

            loader = DataLoader(data, batch_size = args.batch, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
            data_sets.append(loader)

        ## Start experiments
        exp_logger.print("=" * 50 + '\n')
        log = "Experiment: " + str(datetime.datetime.now())
        exp_logger.print(log)
        exp_logger.print(str(args))
        exp_logger.print("="*50+'\n')

        if args.device == "cuda":
            model = torch.load(args.final_model)
        else:
            model = torch.load(args.final_model, map_location=torch.device('cpu'))
        model.to(args.device)

        confusion_matrix = torch.zeros(4, 6, 6)

        exp_logger.print("=" * 50 + '\n')
        exp_logger.print("Testing")
        exp_logger.print("="*50+'\n')

        for i, test_loader in enumerate(data_sets):
            test(test_loader, model, args, confusion_matrix)

        latency = evaluate(data_sets, model, args)
        performance_output(confusion_matrix, latency)


    if args.final == "false":
        version = args.version
        random.seed(args.seed)
        path = args.data_path
        save_path = args.save_path
        groups = os.listdir(path)
        if '.DS_Store' in groups: #remove other files
            groups.remove('.DS_Store')
        print(groups)


        ## Getting training and testing sets
        train_sets_all = []
        test_sets_all = []
        
        #put DataLoader with different groups
        for i in range(len(groups)):
            print(path + groups[i])
            
            if args.new_data == "true":
                train_samples, test_samples, total_cnt = make_dataset(path + groups[i], version, args, groups[i], final_build=args.final)
                # print(test_samples)

                train_data = MyDataset(groups[i], "train", train_samples, test_samples)
                test_data = MyDataset(groups[i], "test", train_samples, test_samples)

            else:
                test_set_verify_g = []
                with open(f'log/test_samples_{groups[i]}.txt', 'r') as file:
                    for line in file:
                        test_set_verify_g.append(ast.literal_eval(line.strip()))


                test_data = MyDataset(groups[i], "test", [], test_set_verify_g)

                train_set_verify_g = []
                with open(f'log/train_samples_{groups[i]}.txt', 'r') as file:
                    for line in file:
                        train_set_verify_g.append(ast.literal_eval(line.strip()))

                train_data = MyDataset(groups[i], "test", [], train_set_verify_g)

            
            train_loader = DataLoader(train_data, batch_size = args.batch_train, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
            test_loader = DataLoader(test_data, batch_size = args.batch_test, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)

            train_sets_all.append(train_loader)
            test_sets_all.append(test_loader)


        # # define_normalization(train_sets_all, test_sets_all)

        ## Start experiments
        exp_logger.print("=" * 50 + '\n')
        log = "Experiment: " + str(datetime.datetime.now())
        exp_logger.print(log)
        exp_logger.print(str(args))
        exp_logger.print("="*50+'\n')

        
        ## Training 1st round: Training swin transformer
        if args.type == "train" or args.type == "all":

            exp_logger.print("=" * 50 + '\n')
            exp_logger.print("Swin Training")
            exp_logger.print("="*50+'\n')

            confusion_matrix = torch.zeros(4, 6, 6)
            num_classes = args.num_classes

            teacher_model = None
            model = models.swin_v2_t(weights='Swin_V2_T_Weights.IMAGENET1K_V1')
            # model = torch.load('./saved_model/' + "swin_iter" + str(version) + ".pkl")
            # model = models.mobilenet_v3_small(weights='MobileNet_V3_Small_Weights.IMAGENET1K_V1')
            # model.classifier = nn.Linear(1000, num_classes)
            # model.classifier = nn.Linear(576, 1000)
            model.to(args.device)

            optimizer = torch.optim.SGD(model.parameters(), args.lr)

            # Define weights of the model
            class_weights = torch.zeros(1000).cuda()
            # Define the new values
            new_values = torch.tensor([15.0, 15.0, 15.0, 15.0, 1.0, 5.0]).cuda()

            weight_log = "Weights: " + str(new_values)
            exp_logger.print(weight_log)

            # Assign the new values to the first 6 elements
            class_weights[:6] = new_values
            class_weights = class_weights.cuda()

            loss_func = torch.nn.CrossEntropyLoss(weight = class_weights).cuda()

            iter = 0
            while iter < args.epochs:
                loss_list = []
                for i, train_loader in enumerate(train_sets_all):
                    
                    loss = train(train_loader, model, teacher_model, loss_func, optimizer, args, confusion_matrix)
                    print("Epoch: ", str(iter), "; iteration: (", str(i), "/", len(train_sets_all), "); loss: ", str(loss))
                    loss_list.append(loss)

                iter += 1
                model_name = './saved_model/' + "swin_final_teacher_iter" + str(version) + ".pkl"
                model_dict_name = './saved_model/' + "swin_final_state_dict_teacher_iter" + str(version) + ".pkl"

                torch.save(model, model_name)
                torch.save(model.state_dict(), model_dict_name)
                if sum(loss_list) / len(loss_list) <= 10:
                    break

                    
            # Save the teacher model
            model_name = './saved_model/' + "swin_final_teacher" + str(version) + ".pkl"
            model_dict_name = './saved_model/' + "swin_final_state_dict_teacher" + str(version) + ".pkl"

            torch.save(model, model_name)
            torch.save(model.state_dict(), model_dict_name)

            # performance_output(confusion_matrix, 0)



        ## Testing the teacher model
        if args.type == "test" or args.type == "all":
            # model = torch.load(model_name)
            if args.device == "cuda":
                model = torch.load('./saved_model/' + "swin_final_teacher" + str(version) + ".pkl")
            else:
                model = torch.load('./saved_model/' + "swin_final_teacher" + str(version) + ".pkl", map_location=torch.device('cpu'))
            # model = torch.load('./saved_model/' + "swin_pretrained2_" + str(version) + ".pkl")
            model.to(args.device)

            confusion_matrix = torch.zeros(4, 6, 6)

            exp_logger.print("=" * 50 + '\n')
            exp_logger.print("Testing")
            exp_logger.print("="*50+'\n')

        
            for i, test_loader in enumerate(test_sets_all):
                test(test_loader, model, args, confusion_matrix)

            latency = evaluate(test_sets_all, model, args)
            # latency = 0
            performance_output(confusion_matrix, latency)





        # Training 2nd round: Knowledge Distillation: transfer knowledge from swin transformer to mobileNet_V3
        if args.type == "train" or args.type == "all":

            exp_logger.print("=" * 50 + '\n')
            exp_logger.print("KD Training")
            exp_logger.print("="*50+'\n')

            confusion_matrix = torch.zeros(4, 6, 6)

            num_classes = args.num_classes

            teacher_model = torch.load('./saved_model/' + "swin_final_teacher" + str(version) + ".pkl")
            teacher_model.to(args.device)
            model = models.mobilenet_v3_small(weights='MobileNet_V3_Small_Weights.IMAGENET1K_V1')
            model.classifier = nn.Linear(576, 1000)
            
            model.to(args.device)

            optimizer = torch.optim.SGD(model.parameters(), args.lr)

            class_weights = torch.zeros(1000).cuda()
            # Define the new values
            new_values = torch.tensor([15.0, 15.0, 15.0, 15.0, 1.0, 5.0]).cuda()

            weight_log = "Weights: " + str(new_values)
            exp_logger.print(weight_log)

            # Assign the new values to the first 6 elements
            class_weights[:6] = new_values
            class_weights = class_weights.cuda()
            loss_func = KnowledgeDistillationLoss(class_weights=class_weights, alpha=0.8).cuda()
        
            iter = 0
            while iter < args.epochs:
                loss_list = []
                for i, train_loader in enumerate(train_sets_all):
                    
                    loss = train(train_loader, model, teacher_model, loss_func, optimizer, args, confusion_matrix)
                    print("Epoch: ", str(iter), "; iteration: (", str(i), "/", len(train_sets_all), "); loss: ", str(loss))
                    loss_list.append(loss)

                iter += 1

                model_name = './saved_model/' + "kd_iter" + str(version) + ".pkl"
                model_dict_name = './saved_model/' + "kd_iter_state_dict" + str(version) + ".pkl"

                torch.save(model, model_name)
                torch.save(model.state_dict(), model_dict_name)
                
                if sum(loss_list) / len(loss_list) <= 10:
                    break

                    
            model_name = './saved_model/' + "kd_final" + str(version) + ".pkl"
            model_dict_name = './saved_model/' + "kd_final_state_dict" + str(version) + ".pkl"

            torch.save(model, model_name)
            torch.save(model.state_dict(), model_dict_name)

            # performance_output(confusion_matrix, 0)


        if args.type == "test" or args.type == "all":
            # model = torch.load(model_name)
            if args.device == "cuda":
                model = torch.load('./saved_model/' + "kd_final" + str(version) + ".pkl")
            else:
                model = torch.load('./saved_model/' + "kd_final" + str(version) + ".pkl", map_location=torch.device('cpu'))
            # model = torch.load('./saved_model/' + "swin_pretrained2_" + str(version) + ".pkl")
            model.to(args.device)

            confusion_matrix = torch.zeros(4, 6, 6)

            exp_logger.print("=" * 50 + '\n')
            exp_logger.print("Testing")
            exp_logger.print("="*50+'\n')


            for i, test_loader in enumerate(test_sets_all):
                test(test_loader, model, args, confusion_matrix)

            latency = evaluate(test_sets_all, model, args)
            # latency = 0
            performance_output(confusion_matrix, latency)



    
        ## Training 1st round: Training swin transformer
        if args.type == "train" or args.type == "all":

            exp_logger.print("=" * 50 + '\n')
            exp_logger.print("MobileNet Training")
            exp_logger.print("="*50+'\n')

            confusion_matrix = torch.zeros(4, 6, 6)
            num_classes = args.num_classes

            teacher_model = None
            model = models.mobilenet_v3_small(weights='MobileNet_V3_Small_Weights.IMAGENET1K_V1')

            # model = models.mobilenet_v3_small()
            model.classifier = nn.Linear(576, 1000)
            # model.load_state_dict(torch.load('./saved_model/ncsu_csc_model_state_dict.pkl'))

            # model = torch.load('./saved_model/' + "swin_iter" + str(version) + ".pkl")
            # model = models.mobilenet_v3_small(weights='MobileNet_V3_Small_Weights.IMAGENET1K_V1')
            # model.classifier = nn.Linear(1000, num_classes)
            model.to(args.device)
            

            optimizer = torch.optim.SGD(model.parameters(), args.lr)

            # Define weights of the model
            class_weights = torch.zeros(1000).cuda()
            # Define the new values
            new_values = torch.tensor([15.0, 15.0, 15.0, 15.0, 1.0, 5.0]).cuda()

            weight_log = "Weights: " + str(new_values)
            exp_logger.print(weight_log)

            # Assign the new values to the first 6 elements
            class_weights[:6] = new_values
            class_weights = class_weights.cuda()

            loss_func = torch.nn.CrossEntropyLoss(weight = class_weights).cuda()

            iter = 0
            while iter < args.epochs:
                loss_list = []
                for i, train_loader in enumerate(train_sets_all):
                    
                    loss = train(train_loader, model, teacher_model, loss_func, optimizer, args, confusion_matrix)
                    print("Epoch: ", str(iter), "; iteration: (", str(i), "/", len(train_sets_all), "); loss: ", str(loss))
                    loss_list.append(loss)

                iter += 1
                model_name = './saved_model/' + "mobilenet_final_iter" + str(version) + ".pkl"
                model_dict_name = './saved_model/' + "mobilenet_final_state_dict_iter" + str(version) + ".pkl"

                torch.save(model, model_name)
                torch.save(model.state_dict(), model_dict_name)
                if sum(loss_list) / len(loss_list) <= 10:
                    break

                    
            # Save the teacher model
            model_name = './saved_model/' + "mobilenet_final" + str(version) + ".pkl"
            model_dict_name = './saved_model/' + "mobilenet_final_state_dict" + str(version) + ".pkl"

            torch.save(model, model_name)
            torch.save(model.state_dict(), model_dict_name)

            # performance_output(confusion_matrix, 0)



        ## Testing the teacher model
        if args.type == "test" or args.type == "all":
            # model = torch.load(model_name)
            if args.device == "cuda":
                model = torch.load('./saved_model/' + "mobilenet_final" + str(version) + ".pkl")
            else:
                model = torch.load('./saved_model/' + "mobilenet_final" + str(version) + ".pkl", map_location=torch.device('cpu'))
            # model = torch.load('./saved_model/' + "swin_pretrained2_" + str(version) + ".pkl")
            model.to(args.device)

            confusion_matrix = torch.zeros(4, 6, 6)

            exp_logger.print("=" * 50 + '\n')
            exp_logger.print("Testing")
            exp_logger.print("="*50+'\n')

        
            for i, test_loader in enumerate(test_sets_all):
                test(test_loader, model, args, confusion_matrix)

            latency = evaluate(test_sets_all, model, args)
            # latency = 0
            performance_output(confusion_matrix, latency)

        
