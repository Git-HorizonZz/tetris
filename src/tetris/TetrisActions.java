package tetris;

import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;

public class TetrisActions implements KeyListener {
	
	final Tetris game;
	
	public TetrisActions(Tetris game) {
		this.game = game;
	}
	
	public void keyTyped(KeyEvent e) { }
		
	public void keyPressed(KeyEvent e) 
	{
		System.out.println(e);
		switch (e.getKeyCode())
		{
			case KeyEvent.VK_UP:
				game.rotate(-1);
				break;
			case KeyEvent.VK_DOWN:
				game.rotate(+1);
				break;
			case KeyEvent.VK_LEFT:
				game.move(-1);
				break;
			case KeyEvent.VK_RIGHT:
				game.move(+1);
				break;
			case KeyEvent.VK_W:
				game.rotate(-1);
				break;
			case KeyEvent.VK_S:
				game.rotate(+1);
				break;
			case KeyEvent.VK_A:
				game.move(-1);
				break;
			case KeyEvent.VK_D:
				game.move(+1);
				break;
			case KeyEvent.VK_SPACE:
				game.drop();
				break;
		} 
	}
	
	public void keyReleased(KeyEvent e) { }
	
}
