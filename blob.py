import numpy as np
import cv2

im = cv2.imread("smile_test.png")
h, w, _ = im.shape
SIZE_x = w
SIZE_y = h

class Blob:
    def __init__(self):
        self.x = np.random.randint(0, w)
        self.y = np.random.randint(0, h)


    def __str__(self):
        #how to show the string
        return f"{self.x}, {self.y}"

    #set the subtraction
    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def action(self, choice):
        agent_plot = []
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)
        # agent_plot.append([self.x, self.y])

    def move(self, x=False, y=False):
        # x=False
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x
        # y=False
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        if self.x < 0:
            self.x = 0
        elif self.x > w -1:
            self.x = w -1
        if self.y < 0:
            self.y = 0
        elif self.y > h -1:
            self.y = h -1