from py4j.java_gateway import JavaGateway

from javaToPython import JavaToPython

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
# javaTalker.get_python_wall()
print(tetris_UI.getWidth())