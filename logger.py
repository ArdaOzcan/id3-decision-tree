import time

from colr import Colr as C


class Logger:
    def __init__(self, app):
        """Construct a Logger object.

        Args:
            app (App): App of this logger.
        """
        self.app = app

    def ask(self, question, prompt="APP"):
        """Call log() with the configured question color and return user input.

        Args:
            question (str): Question to be asked.
            prompt (str, optional): Text that is prompted before time and message. Defaults to 'APP'.
        """

        self.log(question, prompt=prompt,
                 highlight_color=self.app.app_config["colors"]["questionColor"], end='')
        return input()

    def info(self, msg, prompt="APP"):
        """Call log() with the configured info color.

        Args:
            msg (str): Info message to be logged.
            prompt (str, optional): Text that is prompted before time and message. Defaults to 'APP'.
        """

        self.log(msg,
                 self.app.app_config["colors"]["infoColor"],
                 prompt=prompt, end='\n')

    def log(self, msg, highlight_color, prompt='APP', end='\n', bypass_verbose=False):
        """Log message to console with prompt and color.

        Args:
            msg (str): Message to be logged to the console.
            highlight_color (str): Hex value for highlight color.
            prompt (str, optional): Text that is prompted before time and message. Defaults to 'APP'.
            end (str, optional): End of the line. Defaults to '\n'.
            bypass_verbose (bool, optional): Whether the function should care about 'verbose' configuration. Defaults to False.
        """
        if self.app.app_config["booleans"]["verbose"] or bypass_verbose:
            t = time.localtime()
            current_time = time.strftime('%H:%M:%S', t)
            print(C(f"[{prompt}][{current_time}] ",
                    fore=highlight_color, style='bold'), end='')

            # Parts that are inside '$' signs are highlighted with the highlight_color
            parts = msg.split('$')
            special = False
            for i, part in enumerate(parts):
                if special:
                    print(C(part, fore=highlight_color, style='bold'), end='')
                else:
                    print(part, end='')

                special = not special

            # Print newline by default, otherwise the given end value.
            print(end=end)

    def error(self, msg, prompt="APP"):
        """Call log() with the configured error color.

        Args:
            msg (str): Error message to be logged.
            prompt (str, optional): Text that is prompted before time and message. Defaults to 'APP'.
        """
        self.log(msg,
                 self.app.app_config["colors"]["errorColor"],
                 prompt=prompt, bypass_verbose=True)
