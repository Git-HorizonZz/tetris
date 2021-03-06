package tetris;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Point;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;

import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.JPanel;

public class Tetris extends JPanel
{
	private static final long serialVersionUID = 1L;
	private long score = 0;
	private long deltaScore = 0;
	
	public JFrame f = new JFrame("Tetris");
	
	private final Point [] [] [] Tetrominos = 
	{
		{
			{ new Point(0, 1), new Point(1, 1), new Point(2, 1), new Point(3, 1) },		//    []
			{ new Point(1, 0), new Point(1, 1), new Point(1, 2), new Point(1, 3) },		//    []
			{ new Point(0, 1), new Point(1, 1), new Point(2, 1), new Point(3, 1) },		//    []
			{ new Point(1, 0), new Point(1, 1), new Point(1, 2), new Point(1, 3) }		//    []
		},
		{
			{ new Point(0, 1), new Point(1, 1), new Point(2, 1), new Point(2, 0) },		//
			{ new Point(1, 0), new Point(1, 1), new Point(1, 2), new Point(2, 2) },		//    []
			{ new Point(0, 1), new Point(1, 1), new Point(2, 1), new Point(0, 2) },		//    []
			{ new Point(1, 0), new Point(1, 1), new Point(1, 2), new Point(0, 0) }		//  [][]
		},
		{
			{ new Point(0, 1), new Point(1, 1), new Point(2, 1), new Point(2, 2) },		//  
			{ new Point(1, 0), new Point(1, 1), new Point(1, 2), new Point(0, 2) },		//  []
			{ new Point(0, 1), new Point(1, 1), new Point(2, 1), new Point(0, 0) },		//  []
			{ new Point(1, 0), new Point(1, 1), new Point(1, 2), new Point(2, 0) }		//  [][]
		},
		{
			{ new Point(0, 0), new Point(0, 1), new Point(1, 0), new Point(1, 1) },		//
			{ new Point(0, 0), new Point(0, 1), new Point(1, 0), new Point(1, 1) },		//  [][]
			{ new Point(0, 0), new Point(0, 1), new Point(1, 0), new Point(1, 1) },		//  [][]
			{ new Point(0, 0), new Point(0, 1), new Point(1, 0), new Point(1, 1) }		//
		},
		{
			{ new Point(1, 0), new Point(2, 0), new Point(0, 1), new Point(1, 1) },		//    
			{ new Point(0, 0), new Point(0, 1), new Point(1, 1), new Point(1, 2) },		//    [][]
			{ new Point(1, 0), new Point(2, 0), new Point(0, 1), new Point(1, 1) },		//  [][]
			{ new Point(0, 0), new Point(0, 1), new Point(1, 1), new Point(1, 2) }		//
		},
		{
			{ new Point(1, 0), new Point(0, 1), new Point(1, 1), new Point(2, 1) },		//
			{ new Point(1, 0), new Point(0, 1), new Point(1, 1), new Point(1, 2) },		//  [][][]
			{ new Point(0, 1), new Point(1, 1), new Point(2, 1), new Point(1, 2) },		//    []
			{ new Point(1, 0), new Point(1, 1), new Point(2, 1), new Point(1, 2) }		//
		},
		{
			{ new Point(0, 0), new Point(1, 0), new Point(1, 1), new Point(2, 1) },		//
			{ new Point(1, 0), new Point(0, 1), new Point(1, 1), new Point(0, 2) },		//  [][]
			{ new Point(0, 0), new Point(1, 0), new Point(1, 1), new Point(2, 1) },		//    [][]
			{ new Point(1, 0), new Point(0, 1), new Point(1, 1), new Point(0, 2) }		//
		}
	};
	private Point pieceOrigin;
	
	private final Color [] tetrominoColours = 
	{
		Color.cyan, Color.blue, Color.orange, Color.yellow, Color.green, Color.magenta, Color.red
	};
	private Color [] [] wall;
	
	private boolean gameOver = false;
	private boolean episodeOver = false;
	private boolean spawning = false;

	private ArrayList <Integer> next = new ArrayList <Integer>();
	public final int gameWidth = 12;
	public final int gameHeight = 24;
	private int curPiece;
	private int rotation;
	private int county;
	private int [] [] intWall; // wall of 1s and 0s to feed to ai
	ArrayList<Boolean> coveredRows = new ArrayList<Boolean>();

	private double aveY;
	private double aveX;
	
	public Tetris() { }
	
	public void startUp()
	{
		wall = new Color [gameWidth] [gameHeight];
		intWall = new int [gameWidth] [gameHeight];
		for (int i=0; i<gameWidth; i++) 
		{
			for (int k=0; k<gameHeight - 1; k++) 
			{
				// sets border to black
				if (i==0 || i==11 || k==0 || k==22) 
				{
					wall [i] [k] = Color.BLACK;
					intWall [i] [k] = 1;
				}
				else 
				{
					// sets everything else to gray
					wall [i] [k] = Color.GRAY;
					intWall [i] [k] = 0;
				}
			}
		}
		county = 0;
		spawnPiece();
	}
	
	public void spawnPiece()
	{
		spawning = false;
		pieceOrigin = new Point(5, 2);
		if (wall [5][2] != Color.GRAY)
		{
			System.out.println("game over");
			episodeOver = true;
			county++;
			repaint();
		}
		else
		{
			rotation = 0;
			if (next.isEmpty()) 
			{
				Collections.addAll(next, 3);
				Collections.shuffle(next);
			}
			curPiece = next.get(0);
			next.remove(0);
			
			for (Point p : Tetrominos [curPiece] [rotation])
			{
				intWall[pieceOrigin.x + p.x][pieceOrigin.y + p.y] = 1;
			}
			// printWall();
			for (Point p : Tetrominos [curPiece] [rotation])
			{
				intWall[pieceOrigin.x + p.x][pieceOrigin.y + p.y] = 0;
			}
			if (collidesAt(pieceOrigin.x, pieceOrigin.y + 1, rotation)) 
			{
				System.out.println("game over");
				episodeOver = true;
				county++;
				repaint();
			}
		}
	}
	
	private boolean collidesAt(int x, int y, int rotation)
	{
		for (Point p : Tetrominos [curPiece] [rotation]) 
		{
			if (wall [p.x + x] [p.y + y] != Color.GRAY) 
			{
				return true;
			}
		}
		return false;
	}
	
	public boolean rotate(int i)
	{
		boolean canTurn = false;
		int newRotation = (rotation + i) % 4;
		if (newRotation < 0)
		{
			newRotation = 3;
		}
		if (!collidesAt(pieceOrigin.x, pieceOrigin.y, newRotation)) 
		{
			canTurn = true;
			for (Point p : Tetrominos [curPiece] [rotation])
			{
				int row = pieceOrigin.x + p.x;
				intWall [row][pieceOrigin.y + p.y] = 0;
			}
			rotation = newRotation;
			for (Point p : Tetrominos [curPiece] [rotation])
			{
				int row = pieceOrigin.x + p.x;
				intWall [row][pieceOrigin.y + p.y] = 1;
			}
		}
		repaint();
		return canTurn;
	}
	
	public void move(int i)
	{
		if (!collidesAt(pieceOrigin.x + i, pieceOrigin.y, rotation)) 
		{
			for (Point p : Tetrominos [curPiece] [rotation])
			{
				int row = pieceOrigin.x + p.x;
				intWall [row][pieceOrigin.y + p.y] = 0;
			}
			pieceOrigin.x += i;
			for (Point p : Tetrominos [curPiece] [rotation])
			{
				int row = pieceOrigin.x + p.x;
				intWall [row][pieceOrigin.y + p.y] = 1;
			}
		}
		repaint();
	}
	
	public void drop()
	{
		ArrayList<Integer> affectedRows = new ArrayList<Integer>();
		if (!collidesAt(pieceOrigin.x, pieceOrigin.y + 1, rotation)) 
		{
			for (Point p : Tetrominos [curPiece] [rotation])
			{
				int row = pieceOrigin.x + p.x;
				intWall [row][pieceOrigin.y + p.y] = 0;
			}
			pieceOrigin.y++;
			for (Point p : Tetrominos [curPiece] [rotation])
			{
				int row = pieceOrigin.x + p.x;
				intWall [row][pieceOrigin.y + p.y] = 1;
			}
		} 
		else 
		{
			for (Point p : Tetrominos [curPiece] [rotation])
			{
				int row = pieceOrigin.x + p.x;
				boolean newRow = true;

				wall [row][pieceOrigin.y + p.y] = tetrominoColours[curPiece];
				intWall [row][pieceOrigin.y + p.y] = 1;
				
				for(int i : affectedRows){
					if (row == i){
						newRow = false;
					}
				}
				if(newRow)
					affectedRows.add(row);
				
				
			}
			for(int i : affectedRows)
				coveredRows.add(coveredHole(i));
			clearRows();
			spawning = true;
		}	
		repaint();
		// System.out.println("aveY: " + aveY);
	}
	
	public void clearRows() 
	{
		boolean gap;
		int numClears = 0;
		
		for (int k=21; k>0; k--) 
		{
			gap = false;
			for (int i=1; i<11; i++) 
			{
				if (wall [i][k] == Color.GRAY) 
				{
					gap = true;
					break;
				}
			}
			if (!gap) 
			{
				for (int l=k-1; l>0; l--) 
				{
					for (int u=1; u<11; u++) 
					{
						wall [u] [l+1] = wall [u] [l];
						intWall [u] [l+1] = intWall [u] [l];
					}
				}
				k++;
				numClears++;
			}
		}
		deltaScore = 0;
		switch (numClears)
		{
			case 1:
				deltaScore = 100;
				break;
			case 2:
				deltaScore = 300;
				break;
			case 3:
				deltaScore = 500;
				break;
			case 4:
				deltaScore = 800;
				break;
		}

		//resets score if it gets too high
		if(score < Integer.MAX_VALUE / 2)
			score += deltaScore;
		else
		score = 0;
	}
	
	public void paintComponent(Graphics g)
	{
		g.fillRect(0,  0,  312, 598);
		for (int i=0; i<gameWidth; i++) 
		{
			for (int k=0; k<gameHeight - 1; k++) 
			{
				g.setColor(wall [i] [k]);
				g.fillRect(26*i, 26*k, 25, 25);
			}
		}
		g.setColor(Color.WHITE);
		g.setFont(new Font("Courier", Font.PLAIN, 20));
		g.drawString("Score: " + score, 30, 20);
		if (gameOver && county==1)
		{
			// will never run because gameOver is always false
			g.setColor(Color.BLACK);
			g.setFont(new Font("Courier", Font.BOLD, 40));
			g.drawString("GAME OVER", 47, 245);
			GameOver();
		}
		g.setColor(tetrominoColours [curPiece]);
		for (Point p : Tetrominos [curPiece] [rotation]) 
		{
			g.fillRect((p.x + pieceOrigin.x) * 26, (p.y + pieceOrigin.y) * 26, 25, 25);
		}
	}
	
	public void GameOver()
	{
		String options [] = {"Quit"};
		int ans = JOptionPane.showOptionDialog(null, "Score: "+score, "Game Over", JOptionPane.DEFAULT_OPTION, JOptionPane.INFORMATION_MESSAGE, null, options, options[0]);
		if (ans==0)
		{
			System.exit(0);
		}
	}
	
	public boolean getGameOver() 
	{
		return gameOver;
	}
	
	/**
	 * Gets array of where the wall is
	 * @return false if there is no block. true if there is
	 */
	public int[][] getWall()
	{
		int [][] tempWall = intWall;
		return tempWall;
	}

	public byte[] getByteArray(int[][] intArray) {
		int iMax = intArray.length;
		int jMax = intArray[0].length;

		// Set up a ByteBuffer called intBuffer
		ByteBuffer intBuffer = ByteBuffer.allocate(4*iMax*jMax); // 4 bytes in an int
		intBuffer.order(ByteOrder.LITTLE_ENDIAN); // Java's default is big-endian
	
		// Copy ints from intArray into intBuffer as bytes
		for (int i = 0; i < iMax; i++) {
			for (int j = 0; j < jMax; j++){
				intBuffer.putInt(intArray[i][j]);
			}
		}
	
		// Convert the ByteBuffer to a byte array and return it
		byte[] byteArray = intBuffer.array();

		// System.out.println("converted to bytes");
		return byteArray;
	}

	public void printWall()
	{
		for(int row=0; row < wall.length; row++)
		{
			for(int col=wall [0].length-2; col >= 0; col--)
			{
				if (intWall [row][col] == 1)
				{
					System.out.print(" X"); // "X" is printed where a sqaure is covered
				}
				else
				{
					System.out.print(" O"); // "O" is printed where the square is blank
				}
			}
			System.out.println();
		}
		System.out.println();
	}

	public void newEpisode(){
		startUp();
		episodeOver = false;
	}
	
	public int get_X(){
		return pieceOrigin.x;
	}

	public int get_Y(){
		return pieceOrigin.y;
	}

	public int getRotation() 
	{
		// System.out.println(rotation);
		return rotation;
	}

	public byte getWidthBytes(){
		return  Integer.valueOf(gameWidth).byteValue();
	}

	public int getGameHeight(){
		return gameHeight;
	}

	public int getGameWidth(){
		return gameWidth;
	}

	public boolean getColliding(){
		return spawning;
	}

	public void stopColliding(){
		spawning = false;
	}

	public boolean getEpisodeOver(){
		return episodeOver;
	}

	public long getDeltaScore(){
		long dScore = deltaScore;
		// System.out.println("SCORE: " + dScore);
		deltaScore = 0;
		return dScore;
	}

	public boolean canMoveRight()
	{
		return !collidesAt(pieceOrigin.x + 1, pieceOrigin.y, rotation);
	}

	public boolean canMoveLeft()
	{
		return !collidesAt(pieceOrigin.x - 1, pieceOrigin.y, rotation);
	}

	public boolean coveredHole(int row){
		boolean covered = false;
		boolean empty = true;
		int[] intCol = getCol(row);
		// System.out.println("row " + row + ": " + Arrays.toString(intCol));

		for(int i : intCol){
			if (i == 1){
				empty = false;
			} else if (empty == false){
				covered = true;
				// System.out.println("covered");
			}
		}
		return covered;
	}

	public int[] getCol(int colInt) {
		int[] colArr = new int[getWall()[0].length - 2];

		for(int i = 1; i < colArr.length; i++)
			colArr[i] = getWall()[colInt][i];

		return colArr;
	}

	public ArrayList<Boolean> getCoveredRows(){

		ArrayList<Boolean> rows = new ArrayList<Boolean>();

		for(Boolean b : coveredRows){
			rows.add(b);
		}
		coveredRows.clear();

		return rows;
	}

	public double getAveY(){
		aveY = 0;
		for (Point p : Tetrominos [curPiece] [rotation])
		{
			aveY += pieceOrigin.y + p.y;
		}
		aveY /= (double) Tetrominos [curPiece] [rotation].length;
		// System.out.println("java: " + aveY);
		return aveY;
	}

	public double getAveXFromSide(){
		aveX = 0;
		for (Point p : Tetrominos [curPiece] [rotation])
		{
			aveX += 5 - Math.abs(4.5 - pieceOrigin.x + p.x);
		}
		aveX /= (double) Tetrominos [curPiece] [rotation].length;
		aveX = Math.abs(aveX);
		// System.out.println("java: " + aveX);
		return aveX;
	}

	public int getCurrentPiece() {
		return curPiece;
	}
}