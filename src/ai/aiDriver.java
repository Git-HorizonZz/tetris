package ai;

import java.awt.event.KeyEvent;
import java.awt.Robot;
import java.awt.AWTException;

import java.io.*;

import tetris.*;

import org.tensorflow.*;

public class aiDriver {
 
    static Robot robot;

    public aiDriver() throws AWTException{
        robot = new Robot();
    }
    
    public static void main(String[] args) throws IOException, AWTException, InterruptedException
    {
        TetrisDriver.main(args);
        for(int i = 0; i < 100; i++){
        	try {
				Thread.sleep(100);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
            
        }
    }

    public static void moveRight(){
        robot.keyPress(KeyEvent.VK_RIGHT);
    }

    public static void moveLeft(){
        robot.keyPress(KeyEvent.VK_LEFT);
    }

    public static void rotateClockwise(){
        robot.keyPress(KeyEvent.VK_DOWN);
    }

    public static void rotateCounterClockwise(){
        robot.keyPress(KeyEvent.VK_UP);
    }

    public static void dropDown()
    {
        robot.keyPress(KeyEvent.VK_SPACE);
    }

    /**
     * 
     * THINGS AI MUST LEARN
     * 
     * Figure out ideal rotation
     * Figure out ideal position
     * Drop
     * 
     * Store information
     * 
     */

}
