import os

import cv2
import yaml

import utils
from logger import Logger
from node import Node


class App:
    def __init__(self, config_path):
        """Construct an App object.

        Args:
            config_path (str): Path of the config.yaml file.
        """

        self.app_config = yaml.load(open(config_path, "r"),
                                    Loader=yaml.FullLoader)
        self.logger = Logger(self)

    def load_tree(self):
        """Create or read a tree according to config.yaml."""

        root = Node(app=self)
        if self.app_config["booleans"]["calculate"]:
            if root.load_from_csv():
                # If successful
                self.start_id3(root)
            else:
                self.logger.error(
                    'You need to specify a valid .csv file if calculate is set to true')
                return
        else:
            if root.load_from_tree_file():
                # If successful
                img = utils.visualize(root)
                if self.app_config["booleans"]["saveImage"]:
                    base_name = os.path.basename(
                        self.app_config['data']['treeFilePath'])
                    cv2.imwrite(f"{os.path.splitext(base_name[0])}.jpg", img)
            else:
                self.logger.error(
                    'You need to specify a .tree file if calculate is set to false')
                return

    def start_id3(self, root):
        """Create a decision tree with ID3 and make actions according to config.yaml.

        Args:
            root (Node): Root node of the tree.
        """
        root.create_decision_tree_id3()
        img = utils.visualize(root)
        if self.app_config["booleans"]["saveImage"]:
            cv2.imwrite(
                f"{os.path.basename(self.app_config['data']['treeFilePath']).split('.')[0]}.jpg", img)

        if self.app_config["booleans"]["saveTree"]:
            root.save_to_tree_file(
                f"{os.path.basename(self.app_config['data']['csvFilePath']).split('.')[0]}.tree")
