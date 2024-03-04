## imports Path
import nilmtk
from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def extract_states(data):
    whole=[]
    i=0

    for key in data:
        one=0
        zero=0
        for value in key:
            if value==0:
                zero=zero+1
            elif value==1:
                one=one+1
        data = (zero, one)
        whole.append(data)

    zero_values, one_values = zip(*whole)

    return zero_values, one_values

def state_plot(zero_values, one_values):
    categories = ['Fridge', 'Washer Dryer', 'Ketter', 'Dish Washer', 'Microwave']

    # Bar plot
    width = 0.35
    x = np.arange(len(categories))

    percentages_per_pair = [(zero/(zero + one), one / (zero + one))
                            for zero, one in zip(zero_values, one_values)]

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, [pair[0] * 100 for pair in percentages_per_pair], width, label='Zero/Off')
    rects2 = ax.bar(x + width/2, [pair[1] * 100 for pair in percentages_per_pair], width, label='One/On')

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    # Adding percentage values on top of each bar
    def autolabel(rects, percentages):
        for rect, percentage in zip(rects, percentages):
            height = rect.get_height()
            ax.annotate('{:.1f}%'.format(percentage),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1, [pair[0] * 100 for pair in percentages_per_pair])
    autolabel(rects2, [pair[1] * 100 for pair in percentages_per_pair])

    plt.xlabel('Categories')
    plt.ylabel('Counts')
    plt.title('Comparison of Sets')
    plt.show()