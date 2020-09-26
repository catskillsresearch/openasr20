from IPython.display import Audio
import matplotlib.pylab as plt

def show_sound(title, sound, sample_rate):
        print(title)
        display(Audio(sound, rate=sample_rate))
        plt.figure(figsize=(50,8))
        plt.plot(sound);
        plt.xlabel('samples')
        plt.ylabel('amplitude');
        plt.title(title)
        plt.show()
        plt.close()
