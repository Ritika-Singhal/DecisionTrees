import numpy as np


# Information Gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    entropies = []
    attribute_probabilities = []
    for attribute in range(len(branches)):
        total_points = sum(branches[attribute])
        attribute_probabilities.append(float(total_points)/(np.sum(np.array(branches))))
        entropy = sum([(-1)*(float(x)/total_points)*(np.log2(float(x)/total_points)) if x!=0 else 0 for x in branches[attribute]])
        entropies.append(entropy)

    conditional_entropy = sum([x*y for x,y in zip(entropies, attribute_probabilities)])
    return S-conditional_entropy
        

# Implemented reduced error prunning function
def reduced_error_prunning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List
    
    y_pred = decisionTree.predict(X_test)
    correct_count = len([j for i,j in zip(y_test, y_pred) if i==j])
    accuracy = float(correct_count)/len(y_test)
    
    if not decisionTree.root_node.splittable:
        return
    
    all_nodes = []
    
    sub_node = []
    sub_node.append(decisionTree.root_node)
    counter = -1
    while len(sub_node) != 0:
        all_nodes.append(sub_node)
        counter += 1
        sub_node = [[treenode for treenode in n.children if treenode.splittable] for n in all_nodes[counter]]
        sub_node = np.hstack(sub_node).tolist()

    all_nodes = np.hstack(all_nodes).tolist()
    
    for node in reversed(range(0,len(all_nodes))):
        children = all_nodes[node].children
        all_nodes[node].children = []
        all_nodes[node].splittable = False
        
        y_prune_pred = decisionTree.predict(X_test)
        correct_prune_count = len([j for i,j in zip(y_test, y_prune_pred) if i==j])
        accuracy_prune = float(correct_prune_count)/len(y_test)

        if accuracy_prune >= accuracy:
            accuracy = accuracy_prune
        else:
            all_nodes[node].splittable = True
            all_nodes[node].children = children

            
# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')
