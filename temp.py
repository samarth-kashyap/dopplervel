import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    print(f" Enter the main program ")
    A = np.linspace(0, 10, 100)

    plt.figure()
    plt.plot(A, '--r', alpha=0.5, label='test')
    plt.legend()
    plt.show()
