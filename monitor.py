import pygame
import numpy
import random
import torch
from pygame.locals import*
import sys
import os
import numpy as np
import gsmodel

class Monitor:
    def __init__(self):
        pygame.init()                                             # Pygameの初期化
        self.screen = pygame.display.set_mode((600,600))
        self.clock = pygame.time.Clock() # クロックの設定。異なるPCで異なる速さの動作になることを防ぐ
        pygame.display.set_caption("viewer")                        # タイトルバーに表示する文字
        self.visarray=None
    
    def send_4dtensor_2_screen(self,gs_tensor:torch.tensor):
        gs_array=(gs_tensor.numpy()[0,0,:,:]*255).astype("int")
        pixel_size=2
        gs_array=gs_array.repeat(pixel_size,axis=0).repeat(pixel_size,axis=1)
        h=gs_array.shape[0]
        w=gs_array.shape[1]
        screen_arr=pygame.surfarray.pixels3d(self.screen)
        screen_arr[0:w,0:h,0]=gs_array.T[0:w,0:h]
        del screen_arr

    def run(self,gs:gsmodel.GSCott):
        while (1):
            #pygame.surfarray.blit_array(self.screen,np.ones((600,600,3))*255)
            #self.update_screen_1_channel(torch.tensor([[[[0,0,0],[0,0,0],[0,0,0]]]]))
            gs.step()
            self.send_4dtensor_2_screen(gs.u)
            pygame.display.update()     # 画面を更新
            self.clock.tick(40)
            for event in pygame.event.get():
                if event.type == QUIT:  # 閉じるボタンが押されたら終了
                    pygame.quit()       # Pygameの終了(画面閉じられる)
                    sys.exit()

#m=Monitor()
#m.run_single_channel()