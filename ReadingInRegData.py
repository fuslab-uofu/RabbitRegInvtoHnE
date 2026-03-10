import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt


def find_all_numbers(text):
    """
    Finds all numbers (integers, floats, and scientific notation) in a string
    and converts them to float.
    """
    # Regex to match numbers:
    # -? : optional negative sign at the start
    # \d+ : one or more digits
    # \.? : optional decimal point
    # \d* : zero or more digits after the decimal point
    # (?:[Ee][-+]?\d+)? : optional non-capturing group for scientific notation (e/E, optional sign, digits)
    pattern = r"-?\d*\.?\d+(?:[Ee][-+]?\d+)?"

    # Find all matching number strings
    match_number = re.compile(pattern)
    number_strings = re.findall(match_number, text)

    # Convert the extracted strings to float
    # Note: Using float() is the best way to handle all valid numeric formats
    final_list = [float(x) for x in number_strings]

    return final_list


pathypath='/Users/jbonaventura/Desktop/RegDataout1024.xlsx'
df=pd.read_excel(pathypath)
csv_array = df.to_numpy()
print(csv_array.shape)

importantpart=[]
for i in range(csv_array.shape[0]):
    splitbits=csv_array[i,0].split('-')
    if len(splitbits)==6:
        splitbits[4]=splitbits[4]+'-'+splitbits[5]
    importantpart.append(splitbits[4])

#To seperate levels
levelstr=' Registration Complete'
indx=importantpart.index(levelstr)
LevelOne=importantpart[:indx]
TheRest=importantpart[indx+1:]
indx=TheRest.index(levelstr)
LevelTwo=TheRest[:indx]
LevelThree=TheRest[indx+1:]


def SortOutandPlot(LevelList):
    #Sort out numbers-
    Levelnumbs=[]
    for entry in LevelList:
        numbs = find_all_numbers(entry)
        Levelnumbs.append(numbs)

    infolist = []
    for i in range(len(Levelnumbs)):
        length = len(Levelnumbs[i])
        # only do this when length is 1-
        if length == 1:
            its = Levelnumbs[i][0]
            ME = Levelnumbs[i + 1][1]
            RE = Levelnumbs[i + 2][1]
            TE = Levelnumbs[i + 3][1]
            itinfo = [its, ME, RE, TE]
            infolist.append(itinfo)

    infoarray = np.array(infolist)

    fig, axs= plt.subplots(1, 3, figsize=(12, 4))

    axs[0].plot(infoarray[:,0],infoarray[:,1])
    axs[0].set_title("Matching Energy")
    axs[0].set_ylabel("Matching Energy")
    axs[0].set_ylabel("Iterations")

    axs[1].plot(infoarray[:,0],infoarray[:,2])
    axs[1].set_title("Reg Energy")
    axs[1].set_ylabel("Reg Energy")
    axs[1].set_ylabel("Iterations")

    axs[2].plot(infoarray[:,0],infoarray[:,3])
    axs[2].set_title("Total Energy")
    axs[2].set_ylabel("Total Energy")
    axs[2].set_ylabel("Iterations")

    plt.show()

    return infoarray

LevelOneInfo=SortOutandPlot(LevelOne)
LevelTwoInfo=SortOutandPlot(LevelTwo)
LevelThreeInfo=SortOutandPlot(LevelThree)



