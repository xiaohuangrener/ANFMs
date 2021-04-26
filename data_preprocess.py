import time
import pandas as pd
from sklearn.model_selection import train_test_split

class DataPreprocess(object):
    def __init__(self, source_path, target_path, train_path, test_path, validation_path, usecols):
        self.source_path = source_path
        self.target_path = target_path
        self.train_path = train_path
        self.test_path = test_path
        self.validation_path = validation_path
        self.usecols = usecols
        self.users_dict = dict()    #4163
        self.problems_dict = dict() #17751
        self.skills_dict = dict()   #123

        self.read_save()
        self.data_split()

    def read_save(self):
        '''read and map user, problem, skill to continous value,
        (eg. total num_skill=123 in source dataset, but a single skill_id may be greater than 123.)
        save file to assistment.libfm
        '''
        answer_type_dict = {'choose_1': 1, 'choose_n': 2, 'algebra': 3, 'fill_in_1': 4, 'open_response': 5}
        df = pd.read_csv(self.source_path, usecols=self.usecols)
        df.dropna(inplace=True)
        iter_df = df.iterrows()
        count = 1
        with open(self.target_path, 'w') as f:
            for iter in iter_df:
                i, j, k = len(self.users_dict)+1, len(self.problems_dict)+1, len(self.skills_dict)+1
                user = iter[1]['user_id']
                problem = iter[1]['problem_id']
                skill = iter[1]['skill_id']
                correct = iter[1]['correct']
                original = iter[1]['original'] + 1
                answer_type_int = answer_type_dict[iter[1]['answer_type']]
                if user not in self.users_dict:
                    self.users_dict[user] = i
                if problem not in self.problems_dict:
                    self.problems_dict[problem] = j
                if skill not in self.skills_dict:
                    self.skills_dict[skill]  = k

                line = str(int(correct)) + ' ' + str(self.users_dict[user]) + ':1 ' + \
                       str(self.problems_dict[problem] + 4163) + ':1 ' + str(original + 21914) + ':1 ' + \
                       str(answer_type_int + 21916) + ':1 ' + str(int(self.skills_dict[skill]) + 21921) + ':1'

                f.write(line + '\n')
                print('line %d finished'%count)
                count += 1

    def data_split(self):
        '''split assistment.libfm into train, test and validation'''
        with open(self.target_path, 'r') as f, open(self.train_path, 'w') as f_train,\
             open(self.test_path, 'w') as f_test, open(self.validation_path, 'w') as f_val:
            targets = []
            data = []
            line = f.readline()
            while line:
                field = line.strip().split()
                targets.append(field[0])
                data.append(field[1:])
                line = f.readline()

            X_train, X_test, Y_train, Y_test = train_test_split(data, targets, test_size=0.2)
            X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2)
            print("starting save train")
            for i in range(len(Y_train)):
                line = str(Y_train[i]) + ' ' + ' '.join(X_train[i])
                f_train.write(line + '\n')
                print("line %d saved"%i)

            print("starting save validation")
            for i in range(len(Y_val)):
                line = str(Y_val[i]) + ' ' + ' '.join(X_val[i])
                f_val.write(line + '\n')
                print("line %d saved" % i)

            print("starting save test")
            for i in range(len(Y_test)):
                line = str(Y_test[i]) + ' ' + ' '.join(X_test[i])
                f_test.write(line + '\n')
                print("line %d saved" % i)


if __name__ == '__main__':
    source_data_path = "../data/skill_builder_data.csv"
    target_path = "../data/assistment.libfm"
    train_path = "../data/assistment.train.libfm"
    test_path = "../data/assistment.test.libfm"
    validation_path = "../data/assistment.validation.libfm"
    usecols = ['user_id', 'problem_id', 'original', 'correct', 'answer_type', 'skill_id']
    t1 = time.time()
    data = DataPreprocess(source_data_path, target_path, train_path, test_path, validation_path, usecols)
    t2 = time.time()
    print('total time %d'%(t2-t1))
    print(len(data.users_dict))
    print(len(data.problems_dict))
    print(len(data.skills_dict))