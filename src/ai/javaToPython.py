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
        self.terminal.println("J2P")
        byteArray = self.tetris_UI.getByteArray(wall)
        self.terminal.println("got bytes")
        intArray = numpy.frombuffer(byteArray, dtype=numpy.int32)
        self.terminal.println("bytes to ints")
        finalArray = numpy.reshape(intArray, (wall.length, wall[0].length))
        self.terminal.println("2d array")
        return finalArray

    def go_to_location(self, x_pos, rotation):
        self.terminal.println("hello from java talker")
        while self.tetris_UI.getRotation() is not rotation:
            if self.tetris_UI.getRotation() - rotation < 0:
                self.actions_obj.rotateClockwise()
            else:
                self.actions_obj.rotateCounterClockwise()
            time.sleep(0.1)
        
        while self.tetris_UI.getX() != x_pos:
            if self.tetris_UI.getX() > x_pos:
                self.actions_obj.moveLeft()
            else:
                self.actions_obj.moveRight()
            time.sleep(0.1)

