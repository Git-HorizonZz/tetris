import math
import random
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
        self.ave_x = 0
        self.speed = 0.
        self.x_pos = 0
        self.move = 0
        self.not_collided_reward = 0
        self.initial_not_collided_reward = 0.05
        self.just_rotated = False
        
        self.max_pieces = 50
        self.max_rows = 1
        self.max_games = 15
        
        self.total_games = 0
        self.total_pieces = 0
        self.rows_finished = 0
        
    def get_python_wall(self):
        wall = self.tetris_UI.getWall()
        byteArray = self.tetris_UI.getByteArray(wall)
        intArray = numpy.frombuffer(byteArray, dtype=numpy.int32)
        finalArray = numpy.reshape(intArray, (self.tetris_UI.getGameWidth(), self.tetris_UI.getGameHeight()))
        # print(finalArray)
        return intArray
    
    def get_game_over(self):
        if self.tetris_UI.getEpisodeOver():
            self.total_games += 1
            return -1
        elif self.total_pieces > self.max_pieces:
            # print("I win :)")
            self.total_games += 1
            return self.max_pieces * 0.05
        else:
            return 0
        
    def get_episode_over(self):
        return self.rows_finished >= self.max_rows or self.total_games >= self.max_games

    def restart(self, new_episode):
        self.total_pieces = 0
        self.rows_finished = 0
        if new_episode:
            self.total_games = 0
        self.tetris_UI.newEpisode()
        
    def get_reward(self):
        
        rewardY = self.ave_y#(0.1 * self.ave_y ** (1/1.8)) + 0.1
        rewardX = self.ave_x #(0.12 * self.ave_x)
        reward = (0.04 * ((5 * rewardY) - rewardX))
        reward = (1.1 ** reward) - 1.2
        
        
        if self.just_collided():
            self.total_pieces += 1
            # print("collided")
            # print("reward: " + str(self.ave_y))
            
            # print("reward: " + str(reward))
            # print("SCORE: " + str(reward))
            
            score = self.tetris_UI.getDeltaScore()
            if score != 0:
                # print("SCORED!!: " + str(self.tetris_UI.getDeltaScore() / 50))
                reward += (score)
                self.rows_finished += 1
                print("I SCORED!!!!! (" + str(reward) + ")")
                return reward
            else: #if not self.covered_row():
                # print("No cover: " + str(reward))
                return reward
            # else:
            #     # print("cover: " + str(reward / 5))
            #     if reward < 0:
            #         return reward * 4
            #     else:
            #         return reward / 4
        else:
            # if self.not_collided_reward < 0:
            #     print(random.randint(1, 9), end="", flush=True)
            # print("still falling: " + str(self.not_collided_reward))
            return 4 * reward / 5
        
        # testing to see if it can learn anything
        self.just_collided()
        if self.move == 4:
            # print("good" + str(self.move))
            return 1
        else:
            return -0.5
        
            
    
    def just_collided(self):
        if self.tetris_UI.getColliding():
            
            self.ave_y = self.tetris_UI.getAveY()
            self.ave_x = self.tetris_UI.getAveXFromSide()
            # print(self.ave_y)
            self.tetris_UI.spawnPiece()
            return True
        else:
            # self.actions_obj.dropDown()
            # time.sleep(self.speed)
            # self.terminal.println("python: " + str(self.ave_y))
            return False
        
    def covered_row(self):
        covered = False
        for b in self.tetris_UI.getCoveredRows():
            if b:
                covered = True
        return covered
    
    def move_down(self):
        self.actions_obj.moveDown()
        
    def enactAction(self, move):
        action = move.item()
        self.move = move.item()
        self.not_collided_reward = self.initial_not_collided_reward
        if action == 0:
            # if self.just_rotated or not self.actions_obj.rotateClockwise():
            #     self.not_collided_reward = -0.000#6
            self.just_rotated = True
        elif action == 1:
            # if self.just_rotated or not self.actions_obj.rotateCounterClockwise():
            #     self.not_collided_reward = -0.000#6
            self.just_rotated = True
        elif action == 2:
            # if self.tetris_UI.canMoveLeft():
            #     self.not_collided_reward = -0.000#2
            # else:
            #     self.not_collided_reward = 0.0000#7
                # print("bad")
            self.actions_obj.moveLeft()
            self.just_rotated = False
        elif action == 3:
            # if self.tetris_UI.canMoveRight():
            #     self.not_collided_reward = -0.000#2
            # else:
            #     self.not_collided_reward = 0.0000#7
                # print("bad")
            self.actions_obj.moveRight()
            self.just_rotated = False
        elif action == 4:
            self.actions_obj.dropDown()
            self.just_rotated = False
        
    

    def go_to_location(self, position):
        rotation = position // 10
        x_pos = position % 10
        self.x_pos = x_pos
        self.move = position

        rotation = rotation % 4
        rot = rotation.item()
        x = x_pos.item()
        # print("moving")
        while self.tetris_UI.getRotation() is not rot and not self.get_game_over():
            # print("REAL: " + str((self.tetris_UI.getRotation())) + "  GOAL: " + str((rot)))
            if self.tetris_UI.getRotation() - rot < 0:
                self.actions_obj.rotateClockwise()
            else:
                self.actions_obj.rotateCounterClockwise()
            time.sleep(self.speed)
        while self.tetris_UI.get_X() != x and not self.get_game_over():
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

