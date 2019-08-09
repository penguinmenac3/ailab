from typing import List, Optional

from tkinter import *


class Question(object):
    """
    Questions to ask the user.

    You should use the subclasses of question.
    """
    pass


class DropdownQuestion(Question):
    def __init__(self, instruction: str, choices: List[str]) -> None:
        """
       Ask the user a single choice question where he can select from a dropdown menu.

       :param instruction: The instruction what the user should select.
       :param choices: The choices he can make.
       """
        self.instruction = instruction
        self.choices = choices


class TextQuestion(Question):
    def __init__(self, instruction: str) -> None:
        """
       Ask the user a text input question.
       Free text input for the user.

       :param instruction: The instruction what the user should enter.
       """
        self.instruction = instruction


def ask_user(title: str, questions: List[Question]) -> Optional[List[str]]:
    """
   Ask the user a list of questions.

   :param title: Title of the window.
   :param questions: A list of questions that should be asked.
   Questions must be of type DropdownQuestion or TextQuestion.
   :return: The list of answers to the questions.
   """
    root = Tk()
    root.title(title)

    # Add a grid
    mainframe = Frame(root)
    mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
    mainframe.columnconfigure(0, weight=1)
    mainframe.rowconfigure(0, weight=1)
    mainframe.pack(pady=10, padx=10)

    vars = []
    idx = 1
    for question in questions:
        # Create a Tkinter variable
        tkvar = StringVar(root)
        if isinstance(question, DropdownQuestion):
            # Dictionary with options
            tkvar.set(question.choices[0])  # set the default option
            vars.append(tkvar)

            popup_menu = OptionMenu(mainframe, tkvar, *question.choices)
            Label(mainframe, text=question.instruction).grid(row=idx, column=1)
            idx += 1
            popup_menu.grid(row=idx, column=1)
            idx += 1
        elif isinstance(question, TextQuestion):
            raise NotImplementedError("Text questions are not yet implemented.")
        else:
            raise RuntimeError("Only TextQuestion and DropdownQuestion are allowed.")

    confirmed = [False]

    def callback():
        confirmed[0] = True
        root.quit()
        root.destroy()

    Label(mainframe, text="").grid(row=idx, column=1)
    b = Button(mainframe, text="Confirm", command=callback)
    b.grid(row=idx + 1, column=1)

    root.mainloop()

    if confirmed[0]:
        return [tkvar.get() for tkvar in vars]
    else:
        return None


if __name__ == "__main__":
    selected = ask_user("Food Selector",
                        [DropdownQuestion("Select a food", ['Pizza', 'Lasagne', 'Fries', 'Fish', 'Potatoe'])])
    print("{} selected.".format(selected))
