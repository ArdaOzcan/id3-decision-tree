from collections import defaultdict

import pandas

import utils
from logger import info, error


class Node:
    def __init__(self, positive_str=None, categories=None, result=None, parent=None):
        '''Construct a Node object

        Keyword arguments:
        positive_str -- the string that represents a positive outcome
        categories -- columns of data except the result column
        result -- the result column
        parent -- parent of this node if exists.
        '''

        self.children = []
        self.categories = categories
        self.result = result
        self.positive_str = positive_str
        self.value = None
        self.split = None

    def load_from_csv(self, csv_file_name, positive_str):
        prompt = 'READER'
        '''Load data into a Node object from a .csv file

        Positional arguments:
        csv_file_name -- relative or absolute path of the .csv file
        positive_str -- the string that represents a positive outcome, 
                        root node's positive_str is set here.

        '''
        df = pandas.read_csv(csv_file_name)
        info(f"Reading table from csv file $'{csv_file_name}'$...", prompt=prompt)
        self.categories = [(key, df[key].to_list()) for key in df][:-1]
        self.result = (df.iloc[:, -1].name, df.iloc[:, -1].to_list())
        if positive_str not in self.result[1]:
            error(f"Positive value $'{positive_str}'$ was not present in any row of result.", prompt=prompt)
            return False
        self.positive_str = positive_str
        return True

    def create_decision_tree_id3(self):
        prompt = "ID3"
        ''' Create a tree structure that represents a decision tree with the ID3 algorithm. '''

        # If there aren't any categories left, return because all the split operations are done.
        if not self.categories:
            info("Stopped splitting because all attributes were same.", prompt=prompt)
            return

        # Result is the last column of the table and answer of the problem.
        result_title, results = self.result

        ''' 
        Hold entropies for attributes. 
        Structure is : {Category: {Atrb0: H(Atrb0), Atrb1: H(Atrb1)}}
        '''
        entropies = defaultdict(dict)

        '''
        Hold probabilites for attributes 
        to prevent calculating it more than once. 
        Structure is : {Category: {Atrb0: p(Atrb0), Atrb1: p(Atrb1)}}
        '''
        probabilities = defaultdict(dict)

        '''
        Hold the entropy of the result list.
        _ is returned but not used.
        '''
        result_entropy = utils.entropy(results)
        # If result entropy is zero, all results are the same, no need to split.
        if result_entropy == 0.0:
            info("Stopped splitting because all results were same.", prompt=prompt)
            return

        '''
        Loop through all categories to fill entropies and probabilities.
        category -- string that holds the title
        atrb_list -- list that holds the attributes of that category in order.
        '''
        for category, atrb_list in self.categories:
            # Hold results for an attribute in a category
            atrb_results = defaultdict(list)
            # Reduced version of atrb_list, an element occurs at most once.
            atrb_set = set(atrb_list)
            # Hold the length of atrb_list
            atrb_list_len = len(atrb_list)

            # Loop through every attribute that occurs.
            for atrb0 in atrb_set:
                atrb_amount = 0

                # Check the list and count the occurrence amount.
                for i, atrb1 in enumerate(atrb_list):
                    if atrb0 == atrb1:
                        atrb_results[atrb0].append(results[i])
                        atrb_amount += 1

                # Update defaultdicts accordingly
                entropies[category][atrb0]= utils.entropy(atrb_results[atrb0])
                probabilities[category][atrb0] = atrb_amount / atrb_list_len
            info(
                f"Calculated entropies for $'{category}'$", prompt=prompt)
            info(
                f"Calculated probabilities for $'{category}'$", prompt=prompt)

        # Calculate information gain for each category.
        info_gains = defaultdict(float)
        for category, atrb_entropies in entropies.items():
            info_gains[category] = utils.info_gain(
                atrb_entropies, probabilities[category], result_entropy)
            info(
                f"Calculated information gain of $'{category}'$: {info_gains[category]}", prompt=prompt)

        # Find the category that has the maximum information gain.
        max_gain = (self.categories[0][0], 0)
        for category, _ in self.categories:
            if info_gains[category] > max_gain[1]:
                max_gain = (category, info_gains[category])
        info(
            f"Maximum information gain was on $'{max_gain[0]}'$ with ${max_gain[1]}$", prompt=prompt)

        '''
        self.split is shown in visualization, 
        Indicate the category that was split in this node.
        '''
        self.split = max_gain[0]
        info(f"Split the table on $'{max_gain[0]}'$ category", prompt=prompt)

        # Loop through every category in this node.
        for category, parent_atrb_list in self.categories:
            # When you hit max_gain[0], split.
            if category == max_gain[0]:
                '''
                Hold the corresponding indices for every attribute of
                the category with the maximum information gain.
                '''
                child_indices = defaultdict(set)
                for i, value in enumerate(parent_atrb_list):
                    child_indices[value].add(i)

                # Hold the categories field for every child node.
                child_categories = defaultdict(list)
                # Hold the result field for every child node.
                child_results = defaultdict(tuple)

                '''
                Every attribute of the category with the maximum information gain.
                Basically every child that will be split to.
                '''
                for child in set(parent_atrb_list):
                    # Similar to categories but for the result column.
                    child_results[child] = (result_title, [val for i, val in enumerate(
                        results) if i in child_indices[child]])
                    info(
                        f"Results of ${category}->{child}$: {child_results[child][1]}", prompt=prompt)

                    # Loop through every category in this node.
                    for child_category_name, atrb_list in self.categories:
                        '''
                        Append a tuple to the child's category that holds
                        the category name and every attribute that is inside
                        the child's indices.
                        If the category is not the category with the maximum information gain,
                        append values to child categories because max_gain[0] will not be present
                        in children's categories.
                        '''
                        if child_category_name != max_gain[0]:
                            child_categories[child].append((child_category_name, [
                                                           val for i, val in enumerate(atrb_list) if i in child_indices[child]]))

        # Create child nodes and continue recursion.
        for k in child_results.keys():
            t = Node(self.positive_str,
                     categories=child_categories[k], result=child_results[k])
            t.parent = self
            t.value = k
            self.children.append(t)
            t.create_decision_tree_id3()
