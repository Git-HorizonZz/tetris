package ai;
import java.awt.event.KeyEvent;

import tetris.*;

import org.tensorflow.*;

public class aiDriver {
    
    Tetris game = new Tetris();

    public static void main(String[] args) {
        TetrisDriver.main(args);
        for(int i = 0; i < 5; i++){
            TetrisDriver.getActionsObject().keyPressed();
        }
    }
}
