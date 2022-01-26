package ai;

import java.awt.event.KeyEvent;
import java.awt.Robot;
import java.awt.Component;

import tetris.*;

import org.tensorflow.*;

public class aiDriver {

    public static void main(String[] args) {
        TetrisDriver.main(args);
        for(int i = 0; i < 5; i++){
            TetrisDriver.getActionsObject().keyPressed(game);
        }
    }
}
