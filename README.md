# Fcmalgopy
Intuitionistic fuzzy c-means algorithm based on kernels hybridized with a metaheuristic applied to data classification


this is code + research paper on fuzzy c-means algorithms with metaheuristic ( genetic algorithm)



Project contains the following fuzzy c-means algorithms written in python

fcm.py            #Fuzzy c-means clustering algorithm using ecludian distance cost function

kfcm.py           #Kernel fuzzy c-means clustering algorithm using gaussian kernel cost function

ifcm.py           #Intuitionistic fuzzy c-means clustering algorithm using intuitionistic fuzzy similarity/distance metrics (modified ecludian cost function see bellow)

kifcm.py          #Kernel intuitionistic fuzzy c-means clustering algorithm using gaussian kernel cost function


gafcm,gakfcm,gakifcm,gaifcm    #these are all genetic algorithm implementations of each algorithm mentioned above

func.py          #contains PC (Partition Coefficient) and SC (Partition Index) and similarity percentage functions

g.py          #small tkinter ui to display results of each algorithm output




[Research paper.docx](https://github.com/user-attachments/files/15568715/Research.paper.docx)  Research paper containing explination of each algorithm in details written with my uni graduation partner at the time

run g.py      #python ./g.py
