from py4j.java_gateway import JavaGateway

from javaToPython import JavaToPython

import time

'''
Connects to java script
'''
gateway = JavaGateway()
tetris_game = gateway.jvm.tetris.TetrisDriver()
actions_obj = tetris_game.getActionsObject()
tetris_UI = tetris_game.getGameUI()
terminal = gateway.jvm.System.out

javaTalker = JavaToPython(gateway)

terminal.println("hello from python")

while True:
    print(javaTalker.get_episode_over())
    if javaTalker.get_episode_over():
        javaTalker.restart()
    time.sleep(1)