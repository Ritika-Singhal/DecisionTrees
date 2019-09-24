import import_ipynb
import numpy as np
import data
import utils as Util

class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        features_unique = [np.unique(f) for f in np.array(features).T]
        features = np.array(features).tolist()

        # build the tree
        self.root_node = TreeNode(features, labels, features_unique)
        if self.root_node.splittable:
            self.root_node.split()
        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
        return y_pred

class TreeNode(object):
    def __init__(self, features, labels, features_unique):
        # features: List[List[any]], labels: List[int], features_unique: List[List[any]]
        self.features = features
        self.labels = labels
        self.children = []
        self.features_unique = features_unique
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2 or len(np.array(self.features).T) < 1:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    #TODO: try to split current node
    def split(self):
        feature_information_gains = []
        unique_labels, unique_label_count = np.unique(self.labels, return_counts=True)
        
        for f in range(len(np.array(self.features).T)):
            feature_class_count = [[len([i for i,j in zip(np.array(self.features)[:,f], np.array(self.labels)) if i==feature 
                                and j==label]) for label in unique_labels] for feature in self.features_unique[f]]
            Entropy = sum([(-1)*(float(x)/sum(unique_label_count))*np.log2(float(x)/sum(unique_label_count)) 
                                for x in unique_label_count])
            feature_information_gains.append((Util.Information_Gain(Entropy, feature_class_count), len(np.unique(np.array(self.features)[:,f]))))
                      
        information_gains = np.array([i[0] for i in feature_information_gains])
        if all(information_gains==0.0):
            self.splittable = False
            return
        
        self.dim_split = feature_information_gains.index(max(feature_information_gains, key=lambda x: (x[0], x[1])))
        feature_labels = np.column_stack((self.features, self.labels)).tolist()

        feature_labels.sort(key= lambda x: x[self.dim_split])
        feature_unique, unique_index= np.unique(np.array(feature_labels)[:,self.dim_split], return_index=True)
        feature_class_split = np.split(feature_labels, unique_index[1:])
        self.feature_uniq_split = self.features_unique[self.dim_split]

        self.feature_uniq_split = self.feature_uniq_split.tolist()
        self.feature_uniq_split.sort()
        feature_unique = feature_unique.tolist()
        
        for i in range(len(self.feature_uniq_split)):
            if not self.feature_uniq_split[i] in feature_unique:
                new_child = TreeNode([[]], self.labels, [[]])
                new_child.cls_max = self.cls_max
                self.children.append(new_child) 
            else:
                index = feature_unique.index(self.feature_uniq_split[i])   
                child_labels = feature_class_split[index][:,-1]
                child_features = np.delete(feature_class_split[index],-1, 1)
                child_features = np.delete(child_features, self.dim_split, 1)
                child_features_unique = np.delete(self.features_unique, self.dim_split, 0)
                new_child = TreeNode(child_features.tolist(), child_labels.astype(int).tolist(), child_features_unique)
                self.children.append(new_child)
                if new_child.splittable:
                    new_child.split()
        
    # TODO: predict the branch or the class
    def predict(self, feature):
        # feature: List[any]
        # return: int≥
        if not self.splittable:
            return self.cls_max
        else:            
            split_child = self.children[self.feature_uniq_split.index(feature[self.dim_split])]
            feature = np.delete(np.array(feature), self.dim_split)
            return split_child.predict(feature)
