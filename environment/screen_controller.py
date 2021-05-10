import time
from PIL import ImageGrab
import pyautogui

import sys
sys.path.append('./')
from environment import config


class ScreenController():
    def __init__(self):
        self.screen_height = config.screen_height
        self.screen_width = config.screen_width
        self.replayBtn = (self.screen_width/2, self.screen_height/2)

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


controller = ScreenController()
controller.restart()
time.sleep(1)
for i in range(4):
    controller.bow()
    time.sleep(config.default_timestep)
    controller.jump()
    time.sleep(config.default_timestep)
