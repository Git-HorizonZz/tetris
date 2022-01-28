package tetris;

import java.util.Stack;

import javax.swing.JFrame;

import py4j.Gateway;

public class TetrisDriver 
{
	private static int count = 0;
	private static int speed = 1000;
	
	private static final Tetris game = new Tetris();
	private static TetrisActions actions = new TetrisActions(game);
	
	public static void main(String [] args) 
	{
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

	public static TetrisActions getActionsObject() 
	{
		return actions;
	}

	public static Tetris getGameUI()
	{
		return game;
	}

	private Stack stack = new Stack();

    public Stack getStack() {
        return stack;
    }

}
