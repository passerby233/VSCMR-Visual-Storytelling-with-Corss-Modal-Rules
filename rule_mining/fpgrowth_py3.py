"""
This code is adpated from https://github.com/vukk/amdm-fpgrowth-python/blob/master/fpgrowth.py
Must be run in python3, takes 90 minutes.
"""
import sys
import os
import itertools
import argparse
from math import ceil
from collections import defaultdict

import utils as ut
import config as cfg


### USING PYTHON 3 ###

# change these to debug/trace
DEBUG = False
TRACE = False


class FPNode:

    # Node of the FP-tree. Each node has an item, a counter, a parent, a list of
    # childrens and a neighbor
    
    def __init__(self, tree, item): # TODO: tree is not used?
        
        self.item = item
        self.count = 1
        self.parent = None
        self.children = {}
        self.neighbor = None
    
    
    def add(self, child):

        # Add an item
        
        # check if the new item is already one of the childrens of the current
        # node and add it if is not
        if not child.item in self.children:
            self.children[child.item] = child
            child.parent = self
    
    
    def search(self, item):

        # Return the children with item == "item" or None if this doesn't exist
        
        if item in self.children:
            return self.children[item]
        else:
            return None
    
    
    def increment(self):

        # Increment the counter of the current node

        self.count += 1


    def printTree(self, node, transaction_count, minsupport):

        # Print the frequent items of the FP-tree in depth first order
        # Print the items and its counters

        if node:    # recusively traverse tree to print items and their counts and supports
            for child in node.children:
                next = node.children[child]
                if next.count / transaction_count > minsupport:
                    print("|", "Item: ", child, " count: ", next.count, " support: ", next.count / transaction_count)
                self.printTree(next, transaction_count, minsupport)
        else: 
            pass
    
    
    def sortItems(self, node, transaction_count, minsupport, lst):

        # Returns a list of all frequent items in the tree sorted in descending
        # order of support.
        
        if node: # recursively traverse tree to get (node, support) tuples
            for child in node.children:
                next = node.children[child]
                if next.count / transaction_count > minsupport:
                    tup = (child, next.count / transaction_count) # n.count/txcount = support
                    lst.append(tup)
                self.sortItems(next, transaction_count, minsupport, lst)
        else: 
            pass 
            
        # lst is ready...        
        
        # sort lst
        lst = sorted(lst, key=lambda tup: tup[1], reverse = True) # sort by tup[1] which is the support
        return lst


class Route:

    # A route is an element of the routes table.
    # Is a struct with the first and the last node of the route
    def __init__(self):
        
        self.first = None
        self.last = None


class FPTree:

    # FP-tree. Has the tree itself and a table of routes. The table of routes
    # keeps, for each item, the first and the last node which that item.
    # Each node has a pointer to the next node (neighbor). 
    # If there is only one node for a given item, the first and the last of that
    # route is the same node

    def __init__(self):
        
        self.root = FPNode(self, None)
        self.routes = {}
        self.items = list() # set of all items
        self.possibly_frequent = list() # set of possibly still frequent items
        self.transaction_count = 0.0 # float to force floating point divisions

    def mine_frequent_itemsets(self, minsupport, freq_list=list(), freq_list_tuple=list(), prev=list()):
        
        # Mines frequent itemsets by recursively constructing conditional subtrees.
        # Each recursive call finds the k-itemset suffixes that are
        # frequent in the conditional subtree at depth k of recursive calls.
        
        current_freq = list()
        
        for item in self.possibly_frequent:
            trace("is", item, "a frequent item? previous is: ", prev)
            try:
                node = self.routes[item].first
            except KeyError:
                continue # not in routes (not in tree)
            
            sup_count = 0
            
            # find the support count for this item in the current tree
            # by traversing the route of links and summing the counts
            while node != None:
                sup_count += node.count
                node = node.neighbor
            
            # if the item was frequent in this tree, append it to the
            # suffix itemset and add it to the list of frequent itemsets
            if sup_count/self.transaction_count >= minsupport:
                trace(item, "is a frequent item...")
                # prev is the suffix itemset until this point,
                # newprev is the extended/updated suffix itemset
                newprev = list(prev)
                newprev.append(item)
                freq_list.append(newprev)
                freq_list_tuple.append((newprev, sup_count))
                current_freq.append(item)
        
        debug("all frequent on this lvl:", current_freq)
        
        # call gen_prelim_cond_tree() recursively for those items that were found
        # to be frequent (i.e. viable for extending the suffix itemset)
        for item in current_freq:
            prelim_tree = self.gen_prelim_cond_tree(item)
            temp = list(current_freq)
            temp.remove(item)
            prelim_tree.possibly_frequent = temp # update possibly_frequent
                                                 # of the prelim_tree
            
            # call recursively with updated previous-list
            newprev = list(prev)
            newprev.append(item)
            prelim_tree.mine_frequent_itemsets(minsupport, freq_list, freq_list_tuple, newprev)
        
        return (freq_list, freq_list_tuple)

    def traverse_branches_upward_from_leaves(self, item, fun):

        # This traversal calls lambda "fun" with the whole branches of FPNodes

        try:
            leaf = self.routes[item].first
        except KeyError:
            return # item does not exist, just run 0 times
        
        while leaf != None:
            node = leaf
            branch = list()
            
            while node.parent != None:
                branch.append(node)
                node = node.parent
            
            branch.reverse()
            fun(branch) # call lambda with the branch of FPNode objects
        
            leaf = leaf.neighbor

    def gen_prelim_cond_tree(self, item):
        
        # Generate preliminary conditional tree
        # see Florian Verhein:
        #   FP-Growth Algorithm, An Introduction; slides page 7, part 1
        #         (2 slides/page, by part 1 we mean the first slide on the page)
        #   http://csc.lsu.edu/~jianhua/FPGrowth.pdf

        debug("generating prelim cond tree for:", item)
        
        prelim_cond_tree = FPTree()
        prelim_cond_tree.items = self.items
        prelim_cond_tree.transaction_count = self.transaction_count
        
        # function that is called for each branch of FPNodes,
        # branch[-1] is the last item of the list
        # [node.item for node in branch] is a list comprehension to get current branch as items
        fun = lambda branch: prelim_cond_tree.add_actually([node.item for node in branch], branch[-1].count)

        self.traverse_branches_upward_from_leaves(item, fun)
        
        return prelim_cond_tree

    def add(self, transaction, sup_counts, supports, minsupport):
        # Sorts and prunes a transaction so it can be added to the tree
        # sort the transaction from highest to lowest support count
        sorted(transaction, key=lambda item: sup_counts[item], reverse=True)

        # prune infrequent items from the transaction
        pruned_transaction = list()
        for item in transaction:
            if supports[item] >= minsupport:
                pruned_transaction.append(item)

        trace("adding pruned transaction:", pruned_transaction)

        self.add_actually(pruned_transaction)

    def add_actually(self, transaction, addcount = 1):
        # Add a transaction of items to the tree.
        
        point = self.root

        for item in transaction:
            
            next_point = point.search(item) 
            if next_point:
                # If the item is already one of the children, increment its counter
                next_point.increment() 
            else:
                # If is not, create a new node and add it to the tree
                next_point = FPNode(self, item)
                point.add(next_point)
                
                # Add the new node to the routes table
                self.update_route(next_point)
    
            point = next_point  # this is used when constructiong conditional trees
        if addcount > 1:    # if we wish to add greater support upstream than 1
            while point.parent != None:
                point.count += addcount-1
                point = point.parent

    def update_route(self, point):
        # Update the table of routes adding the new item.

        new = False
        
        try:
            temp = self.routes[point.item]
        except KeyError:
        
            # If the item doesn't exist on the routes table,
            # create a new route and it to the table
            
            rt = Route()
            rt.first = point
            rt.last = point
            self.routes[point.item] = rt
            new = True

        if new == False:

            # If the item is already on the routes table,
            # the neighbor of the current last item is the new item. 
            # The new last item is the added item           
        
            self.routes[point.item].last.neighbor = point
            self.routes[point.item].last = point

    def printRoutes(self):
        # Print the table of routes.
        # Print each item of the table and its neighbors.
        
        for rtExt in self.routes:
            rtInt = self.routes[rtExt]
            print("|", rtInt.first.item, " ", end='')
            neighb = rtInt.first.neighbor
            while neighb != None:
                print(neighb.item, " ", end='')
                neighb = neighb.neighbor

            print()

    def generate_association_rules(self, patterns, confidence_threshold):
        """
        Given a set of frequent itemsets, return a dict
        of association rules in the form
        {(left): ((right), confidence)}
        """
        rules = {}
        for itemset in patterns.keys():
            # print "itemset in
            # patterns.keys",itemset,"patterns[itemset]",patterns[itemset]
            upper_support = patterns[itemset]

            for i in range(1, len(itemset)):
                for antecedent in itertools.combinations(itemset, i):
                    antecedent = tuple(sorted(antecedent))
                    consequent = tuple(sorted(set(itemset) - set(antecedent)))
                    if antecedent in patterns:
                        lower_support = patterns[antecedent]
                        confidence = float(upper_support) / lower_support

                        if confidence >= confidence_threshold:
                            rule1 = (consequent, confidence)
                            rule1 = list(rule1)
                            if antecedent in rules:
                                rules[antecedent].append(rule1)
                            else:
                                rules[antecedent] = rule1

        return rules


def show_freqset(freq_tuples):
    # output frequent itemsets
    print("|")
    print("| Frequent itemsets")
    print("| ----------------")
    for (freqset, count) in freq_tuples:
        print(' '.join(freqset), '(' + str(count) + ')')


def convert_to_patterns(freq_tuples):
    patterns = {}
    for freqset, count in freq_tuples:
        patterns[tuple(freqset)] = count
    return patterns


def adaptive_convert_to_patterns(freq_tuples):
    patterns = {}
    for freqsetstr, count in freq_tuples:
        freqset = [int(item) for item in freqsetstr]
        # delete patterns only contain words
        if min(freqset) >= 2048:
            continue
        else:
            patterns[tuple(sorted(freqset))] = count
    return patterns

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--minsupc', type=int, default=3)
    opt = parser.parse_args()

    trans_file = os.path.join(cfg.rule_data_path, "train_transactions.txt")
    pattern_file = os.path.join(cfg.rule_data_path, 'patterns{}.npz'.format(str(opt.minsupc)))

    # if there are not enough arguments given, exit

    # these are the arguments given by the user
    minsupport_c = opt.minsupc
    
    # read the file and construct the FPTree datastructure
    
    fptree = FPTree()
    transaction_count = 0.0  # make it a float, to force floating point divisions
    
    try:
        with open(trans_file, 'r') as f:  # since we use "with", no need to close the file
            sup_counts = defaultdict(int)  # a dictionary with 0 as default

            # 1st pass; count transactions and support counts of items
            for line in f:
                line = line.split()
                for item in line:
                    sup_counts[item] += 1
                transaction_count += 1

            # give fptree some info
            fptree.items = list(sup_counts)  # items in tree = keys of sup_counts
            fptree.possibly_frequent = list(sup_counts)
            fptree.transaction_count = transaction_count
            minsupport = minsupport_c / transaction_count

            # count supports
            supports = {} # is a dictionary
            for i in list(sup_counts):
                supports[i] = sup_counts[i]/transaction_count

            # rewind the data file
            f.seek(0)

            # pass 2, add items to FPTree-datastructure
            for line in f:
                line = line.split()
                fptree.add(line, sup_counts, supports, minsupport)
        
    except IOError:
        sys.exit("File not found")

    # output minsupport, minsupport count and transaction count
    minsupport_count = ceil(transaction_count*minsupport)
    ut.printn(":Finding freq itemset\n",
              "| minsupport:", minsupport,
              "- minsupport count:", minsupport_count,
              "- transaction count:", int(transaction_count))

    if DEBUG:   
        lst = list()
        sorted_list = fptree.root.sortItems(fptree.root, transaction_count, minsupport, lst)
        debug(sorted_list)
        debug()
        debug("Items and counts:")
        debug("-----------------")
        fptree.root.printTree(fptree.root, transaction_count, minsupport) #TODO: RENAME printTree?
        debug()
    
    if DEBUG:
        debug(" Table of routes:")
        debug(" ----------------")
        fptree.printRoutes()
        debug()

    freq_itemsets, freq_tuples = fptree.mine_frequent_itemsets(minsupport)
    # sort by list and length
    freq_tuples = sorted(freq_tuples, key=lambda tup: rec_sort(tup[0]))
    freq_tuples = sorted(freq_tuples, key=lambda tup: len(tup[0]))
    #print(freq_tuples)

    patterns = adaptive_convert_to_patterns(freq_tuples)
    ut.save_dict_npz(pattern_file, patterns)
    ut.printn("Patterns saved to %s" % pattern_file)
    #show_freqset(freq_tuples)


# this is copied from teh internets, deep sort of a list, recursively
def rec_sort(iterable):
    # if iterable is a mutable sequence type
    # sort it
    try:
        iterable.sort()
    # if it isn't return item
    except:
        return iterable
    # loop inside sequence items
    for pos, item in enumerate(iterable):
        iterable[pos] = rec_sort(item)
    return iterable


def debug(*args):
    if DEBUG: print('|', *args) # set if False for no debugging

   
def trace(*args):
    if TRACE: print('|', *args) # set if False for no tracing


if __name__ == '__main__':
    main()

