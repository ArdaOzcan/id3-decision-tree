from app import App

if __name__ == '__main__':
    app = App("config.yaml")
    app.logger.info("Program started.")
    app.load_tree()
