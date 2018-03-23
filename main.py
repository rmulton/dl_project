import sys

from run import launch_training, launch_testing

if __name__ == '__main__':
    args = sys.argv
    if args[1] == 'train':
        if args[2] == "True":
            launch_training(True,args[3],args[4])
        else:
            launch_training(args[2]) #args[2] is input_type parameter
    elif args[1] == 'test':
        launch_testing(args[2],args[3]) #args[3] is input_type parameter
    
