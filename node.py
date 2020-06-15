import json
from collections import defaultdict

import pandas

import utils


class Node:
    def __init__(self, positive_value=None, categories=None, result=None, parent=None, app=None):
        """Construct a Node object.

        Args:
            positive_value (str, optional): The string that represents a positive outcome. Defaults to None.
            categories (list, optional): Columns of data except the result column. Defaults to None.
            result (tuple, optional): The result column. Defaults to None.
            parent (Node, optional): Parent of this node if exists. Defaults to None.
            app (App, optional): App of this node. Defaults to None.
        """

        self.app = app
        self.children = []
        self.categories = categories
        self.result = result
        self.positive_value = positive_value
        self.value = None
        self.split = None

    def load_from_csv(self):
        """Load data into a Node object from a .csv file.

        Returns:
            bool: Whether the loading was successful.
        """
        prompt = 'READER'

        self.csv_file_name = self.app.app_config["data"]["csvFilePath"]
        self.app.logger.info(
            f"Reading table from csv file $'{self.csv_file_name}'$...", prompt=prompt)
        df = pandas.read_csv(self.csv_file_name)

        self.categories = [(key, df[key].to_list()) for key in df][:-1]
        self.result = (df.iloc[:, -1].name, df.iloc[:, -1].to_list())

        if self.app.app_config["data"]["positiveValue"] not in self.result[1]:
            self.app.logger.error(
                f"Positive value $'{self.app.app_config['data']['positiveValue']}'$ was not present in any row of result. " +
                "(Write numbers without quotes in the .yaml file)",
                prompt=prompt)
            return False

        self.positive_value = self.app.app_config['data']['positiveValue']
        return True

    def attributes_of(self, category):
        """Return attribute list of the given category.

        Args:
            category (dict): Given category.

        Returns:
            list: List of attributes in the given category.
        """
        for c, atrb_list in self.categories:
            # When you hit category, return.
            if c == category:
                return atrb_list

    def split_to_children(self):
        """Return child_categories and child_results which are
        used when creating child nodes.

        Returns:
            tuple: Tuple of two dictionaries.
        """
        prompt = 'ID3'

        # Every attribute of the category with the maximum information gain.
        parent_atrb_list = self.attributes_of(self.max_gain[0])

        # Hold the corresponding indices for every attribute of
        # the category with the maximum information gain.
        child_indices = defaultdict(set)
        for i, value in enumerate(parent_atrb_list):
            child_indices[value].add(i)

        # Hold the categories field for every child node.
        child_categories = defaultdict(list)
        # Hold the result field for every child node.
        child_results = defaultdict(tuple)

        # Loop through every child that will be split to.
        for child in set(parent_atrb_list):
            # Similar to categories but for the result column.
            child_results[child] = (self.result[0], [val for i, val in enumerate(
                self.result[1]) if i in child_indices[child]])

            self.app.logger.info(
                f"Results of ${self.max_gain[0]}->{child}$: {child_results[child][1]}", prompt=prompt)

            # Loop through every category in this node.
            for child_category_name, atrb_list in self.categories:

                # Append a tuple to the child's category that holds the category name and
                # every attribute of it whose index is inside the child's indices.
                #
                # If the category is not the category with the maximum information gain,
                # append values to child categories because category with the maximum
                # information gain will not be present in child's categories.
                if child_category_name != self.max_gain[0]:
                    child_categories[child].append((child_category_name, [
                        val for i, val in enumerate(atrb_list) if i in child_indices[child]]))

        return child_categories, child_results

    def calculate_entropies_and_probabilites(self):
        """Calculate entropies ans probabilities for attributes of categories
        and assign them to instance fields.
        """

        prompt = 'ID3'

        # Hold entropies for attributes.
        # Structure is : {Category: {Atrb0: H(Atrb0), Atrb1: H(Atrb1)}}

        self.entropies = defaultdict(dict)

        # Hold probabilites for attributes
        # to prevent calculating it more than once.
        # Structure is : {Category: {Atrb0: p(Atrb0), Atrb1: p(Atrb1)}}
        self.probabilities = defaultdict(dict)

        # Loop through all categories to fill entropies and probabilities.
        # category is a string that holds the title
        # atrb_list is a list that holds the attributes of that category in order.
        for category, atrb_list in self.categories:
            # Hold results for an attribute in a category
            atrb_results = defaultdict(list)
            # Hold the length of atrb_list
            atrb_list_len = len(atrb_list)

            # Loop through every attribute that occurs.
            for atrb0 in set(atrb_list):
                atrb_amount = 0

                # Check the list and count the occurrence amount.
                for i, atrb1 in enumerate(atrb_list):
                    if atrb0 == atrb1:
                        atrb_results[atrb0].append(self.result[1][i])
                        atrb_amount += 1

                # Update defaultdicts accordingly
                self.entropies[category][atrb0] = utils.entropy(
                    atrb_results[atrb0])
                self.probabilities[category][atrb0] = atrb_amount / \
                    atrb_list_len

            self.app.logger.info(
                f"Calculated entropies for $'{category}'$", prompt=prompt)
            self.app.logger.info(
                f"Calculated probabilities for $'{category}'$", prompt=prompt)

    def calculate_info_gains(self):
        """Calculate and return information gain for each category.

        Returns:
            defaultdict: Dictionary of information gains for categories and attributes.
        """

        prompt = 'ID3'
        info_gains = defaultdict(float)
        for category, atrb_entropies in self.entropies.items():
            info_gains[category] = utils.info_gain(
                atrb_entropies, self.probabilities[category], self.result_entropy)
            self.app.logger.info(f"Calculated information gain of $'{category}'$: {info_gains[category]}",
                                 prompt=prompt)

        return info_gains

    def find_max_info_gain(self):
        """Find the category that has the maximum information gain 
        and assign it to instance field.
        """

        prompt = 'ID3'
        self.max_gain = (self.categories[0][0], 0)
        info_gains = self.calculate_info_gains()
        for category, _ in self.categories:
            if info_gains[category] > self.max_gain[1]:
                self.max_gain = (category, info_gains[category])
        self.app.logger.info(f"Maximum information gain was on $'{self.max_gain[0]}'$ with ${self.max_gain[1]}$",
                             prompt=prompt)

    def create_decision_tree_id3(self):
        """Create a tree structure that represents a decision tree with the ID3 algorithm.

        Detailed explanation of the ID3 algorithm: https://en.wikipedia.org/wiki/ID3_algorithm#Algorithm
        Brief explanations are made through block comments.
        """

        prompt = "ID3"

        # If there aren't any categories left, stop recursion
        # because all the split operations are done.
        if not self.categories:
            self.app.logger.info(
                "Stopped splitting because all attributes were same.", prompt=prompt)
            return

        # Result is the last column of the table and answer of the problem.
        result_title, results = self.result

        self.calculate_entropies_and_probabilites()

        # Hold the entropy of the result list.
        self.result_entropy = utils.entropy(results)

        # If result entropy is zero, all results are the same, no need to split.
        # So stop recursion.
        if self.result_entropy == 0.0:
            self.app.logger.info(
                "Stopped splitting because all results were same.", prompt=prompt)
            return

        self.find_max_info_gain()

        # Indicate the category that was split in this node.
        self.split = self.max_gain[0]
        self.app.logger.info(f"Split the table on $'{self.max_gain[0]}'$ category",
                             prompt=prompt)

        # Create child nodes and continue recursion.
        child_categories, child_results = self.split_to_children()
        for child in child_results.keys():
            t = Node(self.positive_value,
                     categories=child_categories[child],
                     result=child_results[child],
                     parent=self,
                     app=self.app)
            t.value = child
            self.children.append(t)
            # Continue recursion with child
            t.create_decision_tree_id3()

    def save_to_tree_file(self, file_name):
        """Write the tree whose root is this node to a JSON file.

            NOTE: Loaded information does not contain numeric values or
                  attributes except the split attribute and the value, 
                  neither does the loaded information. These information 
                  are only used to access or visualize the resulting tree.

        Args:
            file_name (str): Name of file to be written to.
        """
        with open(file_name, "w") as file:
            json.dump(self.get_node_data(), file)

    def load_from_data(self, data):
        """Load the node and its children from a dictionary.

        This method is recursive and is used to form a tree.
        First call to this method contains information about the whole
        tree. Information reduces as the method goes further in recursion.

            NOTE: Loaded information does not contain numeric values or
                  attributes except the split attribute and the value, 
                  neither does the saved information. These information 
                  are only used to access or visualize the resulting tree.

        Args:
            data (dict): Information about this node and its children.
        """

        self.value = data["value"]
        self.split = data["split"]
        self.result = tuple(data["result"])
        self.positive_value = data["positive_value"]
        for c in data["children"]:
            child = Node(parent=self)
            # Continue recursion
            child.load_from_data(c)
            self.children.append(child)

    def load_from_tree_file(self):
        """Load a tree whose root is this node from a JSON file.

        JSON file read with this method must belong to the root node,
        otherwise the tree will start from whatever node you called
        the save method from.

        Returns:
            bool: Whether the loading was successful.
        """
        try:
            with open(self.app.app_config["data"]["treeFilePath"], "r") as file:
                self.load_from_data(json.load(file))
            return True
        except (FileNotFoundError, TypeError):
            return False

    def get_node_data(self):
        """Form and return some information about this node and its children.

        This method is used for recursively writing the tree to a JSON file.

        Returns:
            dictionary: Some information about this node and its children.
        """
        return {"value": self.value,
                "split": self.split,
                "result": self.result,
                "positive_value": self.positive_value,
                "children": [c.get_node_data() for c in self.children]}
