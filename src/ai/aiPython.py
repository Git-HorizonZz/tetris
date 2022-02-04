from py4j.java_gateway import JavaGateway

import numpy

import time

'''
Connects to java script
'''
gateway = JavaGateway()
tetris_game = gateway.jvm.tetris.TetrisDriver()
actions_obj = tetris_game.getActionsObject()
tetris_java = tetris_game.getGameUI()
terminal = gateway.jvm.System.out



# numpy.zeros((state_size, 5))

while True:
    terminal.print(tetris_java.getX())
    time.sleep(0.1)