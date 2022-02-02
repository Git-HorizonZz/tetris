from py4j.java_gateway import JavaGateway

import time

'''
Connects to java script
'''
gateway = JavaGateway()
tetris_game = gateway.jvm.tetris.TetrisDriver()
actions_obj = tetris_game.getActionsObject()

while True:
    actions_obj.moveRight()
    time.sleep(0.1)