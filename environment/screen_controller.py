import time
from PIL import ImageGrab, ImageOps, ImageDraw, Image
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
        self.replay_btn = env_config.replay_btn

    def restart(self):
        pyautogui.click(self.replay_btn)

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
        gray_image = gray_image.resize(env_config.obs_shape)
        #print(gray_image.size)
        #gray_image.show()
        array_image = np.array(gray_image)
        # convert_back_image = Image.fromarray(array_image)
        # convert_back_image.show()
        # print(convert_back_image.size)
        return array_image


# controller = ScreenController()
# img = controller.screen()
# controller.restart()
# time.sleep(1)
# for i in range(4):
#     controller.bow()
#     time.sleep(env_config.default_timestep)
#     controller.jump()
#     time.sleep(env_config.default_timestep)
