def progress_bar(fig, ax, progress):
    ax.plot(progress)
    fig.canvas.draw()
