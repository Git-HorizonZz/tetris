from py4j.java_gateway import JavaGateway

import numpy
from numpy import array

import time

class JavaToPython():

    def __init__(self, gateway):
        self.gateway = gateway
        self.tetris_game = self.gateway.jvm.tetris.TetrisDriver()
        self.actions_obj = self.tetris_game.getActionsObject()
        self.tetris_UI = self.tetris_game.getGameUI()
        self.terminal = self.gateway.jvm.System.out

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
        if self.tetris_UI.getDeltaScore() != 0:
            return self.tetris_UI.getDeltaScore() / 200
        else:
            return 0.05
    
    def just_collided(self):
        print(self.tetris_UI.getColliding())
        if self.tetris_UI.getColliding():
            self.tetris_UI.stopColliding()
            return True
        else:
            self.terminal.println("not collided")
            self.actions_obj.dropDown()
            return False

    def go_to_location(self, position):
        rotation = position // 10
        x_pos = position % 10
        print(position)
        print(rotation)
        print(x_pos)

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
            time.sleep(0.1)
        while self.tetris_UI.getX() != x and not self.get_episode_over():
            self.terminal.println("REAL: " + str((self.tetris_UI.getX())) + "  GOAL: " + str((x)))
            if self.tetris_UI.getX() > x:
                if not self.tetris_UI.canMoveLeft():
                    self.terminal.println("correct pos")
                    return
                self.actions_obj.moveLeft()
            else:
                if not self.tetris_UI.canMoveRight():
                    # print("correct pos")
                    return
                self.actions_obj.moveRight()
            time.sleep(0.1)
        self.terminal.println("correct pos")

