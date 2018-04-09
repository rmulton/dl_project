import sys

from run import launch_training, launch_testing

if __name__ == '__main__':
    args = sys.argv
    print(args)
    if args[1] == 'train':
        if args[2] == "True":
            launch_training(True,int(args[3]),args[4]) #args[3] is the input_type parameter
        else:
            launch_training(False,int(args[3])) #args[3] is input_type parameter
    elif args[1] == 'test':
        launch_testing(args[2],int(args[3])) #args[3] is input_type parameter
    
