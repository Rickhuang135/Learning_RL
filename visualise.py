from matplotlib import pyplot as plt

def losses(training: list, validation: list):
    plt.plot(training, label="training losses")
    plt.plot(validation, label="validation losses")
    plt.legend()
    plt.title("Loss over epochs")
    plt.show()