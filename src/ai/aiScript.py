from py4j.java_gateway import JavaGateway

from javaToPython import JavaToPython

'''
Connects to java script
'''
gateway = JavaGateway()
tetris_game = gateway.jvm.tetris.TetrisDriver()
actions_obj = tetris_game.getActionsObject()
tetris_java = tetris_game.getGameUI()
terminal = gateway.jvm.System.out

java = JavaToPython(gateway)

terminal.println("hello from python")
java.go_to_location(1, 2)