import math
from random import random
from py4j.java_gateway import JavaGateway

import numpy
from numpy import array

from random import randint

import time

class JavaToPython():

    def __init__(self, gateway):
        self.gateway = gateway
        self.tetris_game = self.gateway.jvm.tetris.TetrisDriver()
        self.actions_obj = self.tetris_game.getActionsObject()
        self.tetris_UI = self.tetris_game.getGameUI()
        self.terminal = self.gateway.jvm.System.out
        
        self.ave_y = 0
        self.speed = 0.000
        self.x_pos = 0
        self.move = 0
        
    def get_python_wall(self):
        wall = self.tetris_UI.getWall()
        byteArray = self.tetris_UI.getByteArray(wall)
        intArray = numpy.frombuffer(byteArray, dtype=numpy.int32)
        finalArray = numpy.reshape(intArray, (self.tetris_UI.getGameWidth(), self.tetris_UI.getGameHeight()))
        return intArray
    
    def get_episode_over(self):
        return self.tetris_UI.getEpisodeOver()

    def restart(self):
        self.tetris_UI.newEpisode()
        
    def get_reward(self):
        # # print("reward: " + str(self.ave_y))
        # reward = (0.08 * self.ave_y ** (1/1.8)) - 0.
        # # print("SCORE: " + str(reward))
        # if self.tetris_UI.getDeltaScore() != 0:
        #     # print("SCORED!!: " + str(self.tetris_UI.getDeltaScore() / 50))
        #     return self.tetris_UI.getDeltaScore() / 50
        # elif not self.covered_row():
        #     # print("No cover: " + str(reward))
        #     return reward
        # else:
        #     # print("cover: " + str(reward / 5))
        #     return reward / 5
        
        if self.move % 2 == 0 and self.tetris_UI.getCurrentPiece() == 3:
            # print("good" + str(self.move))
            return 1
        elif self.move % 2 == 1 and self.tetris_UI.getCurrentPiece() == 4:
            return 1
        else:
            # print("bad" + str(self.move))
            return -0.5
        
            
    
    def just_collided(self):
        if self.tetris_UI.getColliding():
            
            self.ave_y = self.tetris_UI.getAveY()
            # print(self.ave_y)
            self.tetris_UI.spawnPiece()
            return True
        else:
            self.actions_obj.dropDown()
            time.sleep(self.speed)
            # self.terminal.println("python: " + str(self.ave_y))
            return False
        
    def covered_row(self):
        covered = False
        for b in self.tetris_UI.getCoveredRows():
            if b:
                covered = True
        return covered
        

    def go_to_location(self, position):
        rotation = position // 10
        x_pos = position % 10
        self.x_pos = x_pos
        self.move = position

        rotation = rotation % 4
        rot = rotation.item()
        x = x_pos.item()
        # print("moving")
        while self.tetris_UI.getRotation() is not rot and not self.get_episode_over():
            # print("REAL: " + str((self.tetris_UI.getRotation())) + "  GOAL: " + str((rot)))
            if self.tetris_UI.getRotation() - rot < 0:
                self.actions_obj.rotateClockwise()
            else:
                self.actions_obj.rotateCounterClockwise()
            time.sleep(self.speed)
        while self.tetris_UI.get_X() != x and not self.get_episode_over():
            # self.terminal.println("REAL: " + str((self.tetris_UI.get_X())) + "  GOAL: " + str((x)))
            if self.tetris_UI.get_X() > x:
                if not self.tetris_UI.canMoveLeft():
                    # self.terminal.println("correct pos")
                    return
                self.actions_obj.moveLeft()
            else:
                if not self.tetris_UI.canMoveRight():
                    # print("correct pos")
                    return
                self.actions_obj.moveRight()
            time.sleep(self.speed)
        # self.terminal.println("correct pos")

