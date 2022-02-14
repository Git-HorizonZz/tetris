from py4j.java_gateway import JavaGateway

import numpy

import time

class JavaToPython():

    def __init__(self, gateway):
        self.gateway = gateway
        self.tetris_game = self.gateway.jvm.tetris.TetrisDriver()
        self.actions_obj = self.tetris_game.getActionsObject()
        self.tetris_java = self.tetris_game.getGameUI()
        self.terminal = self.gateway.jvm.System.out


    def go_to_location(self, x_pos, rotation):
        while self.tetris_java.getRotation() is not rotation:
            if self.tetris_java.getRotation() - rotation < 0:
                self.actions_obj.rotateClockwise()
            else:
                self.actions_obj.rotateCounterClockwise()
            time.sleep(0.1)
        
        while self.tetris_java.getX() != x_pos:
            if self.tetris_java.getX() > x_pos:
                self.actions_obj.moveLeft()
            else:
                self.actions_obj.moveRight()
            time.sleep(0.1)

