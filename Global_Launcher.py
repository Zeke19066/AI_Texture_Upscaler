import Resize
import multiprocessing


#c:\Users\Ezeab\Documents\Python\Trading_Agent\Trader_NN.py:812: UserWarning: nn.init.uniform is now deprecated in favor of nn.init.uniform_.
#  torch.nn.init.uniform(m.weight, -0.01, 0.01)

#To terminate, type "Terminate"

def main(mode):
    i = 0
    try:
        multiprocessing.set_start_method("spawn")
        #recieve_connection, send_connection = multiprocessing.Pipe(duplex=False)
        mode = 'batch'
        #nn_process = multiprocessing.Process(target=Neural_Network.main, args=(mode, send_connection, first_run, ticker))
        #gui_process = multiprocessing.Process(target=Trader_GUI.main, args=(recieve_connection, ticker))
        resize_process = multiprocessing.Process(target=Resize.main, args=(mode))
        resize_process.start()

    except:
        resize_process.terminate()
        i+=1
        print(f"something went wrong count:{i}")
        return


if __name__ == "__main__":
    main(mode="Train")