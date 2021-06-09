import time
from PIL import ImageGrab, ImageOps, ImageDraw
import pyautogui
import numpy as np

import sys
sys.path.append('./')
from environment import env_config


class ScreenController():
    def __init__(self):
        self.screen_height = env_config.screen_height
        self.screen_width = env_config.screen_width
        self.screen_start = env_config.screen_start
        self.replayBtn = (self.screen_start[0] + self.screen_width/2, self.screen_start[1] + 260)

    def restart(self):
        pyautogui.click(self.replayBtn)

    def jump(self):
        pyautogui.keyDown('space')
        time.sleep(0.05)
        pyautogui.keyUp('space')

    def bow(self):
        pyautogui.keyDown('down')
        time.sleep(0.05)
        pyautogui.keyUp('down')

    def screen(self):
        box = (self.screen_start[0], self.screen_start[1], self.screen_start[0]+self.screen_width , self.screen_start[1]+self.screen_height)
        image = ImageGrab.grab(box)
        gray_image = ImageOps.grayscale(image)
        # gray_image.show()
        array_image = np.array(gray_image)
        return array_image


# controller = ScreenController()
# controller.screen()
# controller.restart()
# time.sleep(1)
# for i in range(4):
#     controller.bow()
#     time.sleep(config.default_timestep)
#     controller.jump()
#     time.sleep(config.default_timestep)
