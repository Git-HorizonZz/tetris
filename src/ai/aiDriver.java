package ai;

import java.awt.event.KeyEvent;

import tetris.*;

import org.tensorflow.*;

public class aiDriver {

    public static void main(String[] args) {
        TetrisDriver.main(args);
        for(int i = 0; i < 100; i++){
        	try {
				Thread.sleep(100);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
            TetrisDriver.getActionsObject().keyPressed(leftKey());
        }
    }

    public static KeyEvent rightKey(){
        return new KeyEvent(TetrisDriver.getGameUI().f, KeyEvent.KEY_LOCATION_RIGHT, System.currentTimeMillis(), 0, KeyEvent.VK_RIGHT, KeyEvent.CHAR_UNDEFINED);
    }

    public static KeyEvent leftKey(){
        return new KeyEvent(TetrisDriver.getGameUI().f, KeyEvent.KEY_LOCATION_LEFT, System.currentTimeMillis(), 0, KeyEvent.VK_LEFT, KeyEvent.CHAR_UNDEFINED);
    }

    public static KeyEvent upKey(){
        return new KeyEvent(TetrisDriver.getGameUI().f, KeyEvent.VK_KP_DOWN, System.currentTimeMillis(), 0, KeyEvent.VK_UP, KeyEvent.CHAR_UNDEFINED);
    }

    public static KeyEvent downKey(){
        return new KeyEvent(TetrisDriver.getGameUI().f, KeyEvent.VK_KP_UP, System.currentTimeMillis(), 0, KeyEvent.VK_DOWN, KeyEvent.CHAR_UNDEFINED);
    }
}
