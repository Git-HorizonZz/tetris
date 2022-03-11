package tetris;

import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.Robot;

import java.awt.AWTException;



public class TetrisActions implements KeyListener {
	
	final Tetris game;
	static Robot robot;

	
	public TetrisActions(Tetris game) throws AWTException
	{
		robot = new Robot();
		this.game = game;
	}
	
	public void keyTyped(KeyEvent e) { }
		
	public void keyPressed(KeyEvent e) 
	{
		switch (e.getKeyCode())
		{
			case KeyEvent.VK_SPACE:
				game.drop();
				break;
		} 
	}
	
	public void keyReleased(KeyEvent e) { }
	
	public void moveRight()
    {
        game.move(+1);
    }

    public void moveLeft()
    {
        game.move(-1);
    }

    public void rotateClockwise()
    {
        game.rotate(+1);
    }

    public void rotateCounterClockwise()
    {
		game.rotate(-1);
    }

    public void dropDown()
    {
        game.drop();
    }

}
