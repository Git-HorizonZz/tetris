package tetris;

import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;

import javax.swing.JFrame;
import javax.swing.JOptionPane;

public class TetrisDriver 
{
	private static int count = 0;
	private static int speed = 1000;
	
	public static void main(String [] args) 
	{
		JOptionPane.showMessageDialog(null, "rotate left : w, up \nrotate right : s, down \nmove left : a, left\nmove right : d, right\ndrop : space", "Controls", JOptionPane.DEFAULT_OPTION);
		JOptionPane.showMessageDialog(null, "1 row : 100 points\n2 rows : 300 points\n3 rows : 500 points\n4rows : 800 points", "Points", JOptionPane.DEFAULT_OPTION);
		Tetris tts = new Tetris();
		JFrame f = tts.f;
		f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		f.setSize(310, 623);
		final Tetris game = new Tetris();
		game.startUp();
		f.add(game);
		f.setLocationRelativeTo(null);
		f.setVisible(true);
		f.setResizable(false);
		
		f.addKeyListener(new KeyListener() 
		{
			public void keyTyped(KeyEvent e) { }
			
			public void keyPressed(KeyEvent e) 
			{
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
		});
		
		new Thread() 
		{
			public void run() 
			{
				while (true) 
				{
					try 
					{
						Thread.sleep(speed);
						game.drop();
						count++;
						if (count % 5 == 0 && speed > 200)
						{
							speed-=10;
						}
					} 
					catch (InterruptedException e) { }
				}
			}
		}.start();
	}
}
