class LoadData(object):
    def __init__(self, assistment_file, train_file, test_file, val_file):
        self.assistment_file = assistment_file
        self.train_file = train_file
        self.test_file = test_file
        self.val_file = val_file
        self.features = {}
        self.num_features = self.count_features_num(self.assistment_file)
        self.train_data, self.test_data, self.val_data = self.construct_data()

    def count_features_num(self, file):
        with open(file, 'r') as f:
            line = f.readline()
            while line:
                items = line.strip().split()
                for item in items[1:]:
                    if item not in self.features:
                        self.features[item] = len(self.features)
                line = f.readline()
        return len(self.features)

    def construct_data(self):
        train_data = self.read_data(self.train_file)
        test_data = self.read_data(self.test_file)
        val_data = self.read_data(self.val_file)
        return train_data, test_data, val_data

    def read_data(self, file):
        data = {}
        X_ = []
        Y_ = []
        with open(file, 'r') as f:
            line = f.readline()
            while line:
                items = line.strip().split()
                Y_.append(items[0])
                X_.append([self.features[item] for item in items[1:]])
                line = f.readline()
            data['X'] = X_
            data['Y'] = Y_
        return data