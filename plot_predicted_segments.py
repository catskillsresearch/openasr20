from matplotlib.pylab import *

def plot_predicted_segments(timeline, normalized_power, speech_mask, segment_transcript, gold):
    figure(figsize=(24,6))
    plot(timeline, normalized_power)
    plot(timeline, speech_mask, color='green')
    segment_begin=segment_transcript[0][0]
    segment_end=segment_transcript[-1][0]
    for x, tdur, word in segment_transcript:
        plot([x, x+tdur], [0, 0], color='magenta',linewidth=3)
        text(x, 1.1, word, color='blue')
        axvline(x=x,color='yellow')
    for y, phrase in gold:
        if y < segment_begin:
            continue
        if y > segment_end:
            break
        text(y, 1.2, phrase, color='brown')
        axvline(x=y,color='orange')
        
