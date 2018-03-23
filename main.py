import sys

from run import launch_training, launch_testing

if __name__ == '__main__':
    args = sys.argv
    if args[1] == 'train':
        if args[2] == True:
            launch_training(True,args[3])
        else:
            launch_training()
    elif args[1] == 'test':
        launch_testing(args[2])
    
