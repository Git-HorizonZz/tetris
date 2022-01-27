package tetris;

import javax.swing.JFrame;
import javax.swing.JOptionPane;

public class TetrisDriver 
{
	private static int count = 0;
	private static int speed = 1000;
	
	private static final Tetris game = new Tetris();
	private static TetrisActions actions = new TetrisActions(game);
	
	public static void main(String [] args) 
	{
		JOptionPane.showMessageDialog(null, "rotate left : w, up \nrotate right : s, down \nmove left : a, left\nmove right : d, right\ndrop : space", "Controls", JOptionPane.DEFAULT_OPTION);
		JOptionPane.showMessageDialog(null, "1 row : 100 points\n2 rows : 300 points\n3 rows : 500 points\n4rows : 800 points", "Points", JOptionPane.DEFAULT_OPTION);
		Tetris tts = new Tetris();
		JFrame f = tts.f;
		f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		f.setSize(310, 623);
		
		game.startUp();
		f.add(game);
		f.setLocationRelativeTo(null);
		f.setVisible(true);
		f.setResizable(false);
		
		f.addKeyListener(new TetrisActions(game));
		
		new Thread() 
		{
			public void run() 
			{
				while (!game.getGameOver()) 
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

	public static TetrisActions getActionsObject() {
		return actions;
	}

	public static Tetris getGameUI(){
		return game;
	}
}
