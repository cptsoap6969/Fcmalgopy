from threading import Thread
import numpy as np
import tkinter as tk
from tkinter import ttk , Text , END , Scrollbar , RIGHT
from sklearn.datasets import load_iris , load_wine , load_breast_cancer
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import ifcm , fcm ,kfcm ,gaifcm , gakfcm , gafcm, kifcm, gakifcm , func
import pandas as pd

pattern_done = False

def run_algorithm():
    selected_algorithm = algorithm_var.get()
    selected_dataset = dataset_var.get()
    input1 = input_one.get()
    input2 = input_two.get()
    input3 = input_three.get()
    input4 = input_four.get()
    input5 = input_pop.get()
    input6 = input_gen.get()
    global graph , c , X,  weight, distance_matrix, m, perf,C,distance_centers
    if selected_dataset == "Iris":
        X = load_iris().data
        F = load_iris().target
        alfa = 1.27
        c = 3
    elif selected_dataset == "Wine":
        X = load_wine().data
        F = load_wine().target
        alfa = 150
        c = 3
    elif selected_dataset == "Breast Cancer":
        X = load_breast_cancer().data
        F = load_breast_cancer().target
        alfa = 150
        c = 2
    elif selected_dataset == "Seeds":
        X = pd.read_csv("seeds.csv").drop(['V8'], axis=1).values
        F = pd.read_csv("seeds.csv")['V8'].values
        alfa = 150
        c = 3
    elif selected_dataset == "Dermatology":
        X = pd.read_csv("derm.csv").drop(['class'], axis=1).values
        F = pd.read_csv("derm.csv")['class'].values
        alfa = 150
        c = 6

    if selected_algorithm == 'Ifcm':
        graph , c , X,  weight, distance_matrix, m, perf , C,distance_centers= ifcm.graph_init(X , c)
        update_plot(graph ,c ,X , func.PC(weight),perf,C, func.SC(weight,c,distance_matrix, m,distance_centers))
    elif selected_algorithm == 'Kifcm':
        graph , c , X,  weight, distance_matrix, m, perf , C,distance_centers = kifcm.graph_init(X , c, alfa)
        update_plot(graph ,c ,X , func.PC(weight), perf,C, func.SC(weight,c,distance_matrix, m,distance_centers))
    elif selected_algorithm == 'Fcm':
        graph , c , X,  weight, distance_matrix, m, perf , C,distance_centers= fcm.graph_init(X , c)
        update_plot(graph ,c ,X , func.PC(weight), perf,C,func.SC(weight,c,distance_matrix, m,distance_centers))
    elif selected_algorithm == 'Kfcm':
        graph , c , X,  weight, distance_matrix,m , perf , C ,distance_centers= kfcm.graph_init(X , c,alfa)
        update_plot(graph ,c ,X , func.PC(weight),perf ,C,func.SC(weight,c,distance_matrix, m,distance_centers))
    elif selected_algorithm == 'GAifcm':
        graph , c , X, weight, distance_matrix, m, lamda, perf , C,distance_centers = gaifcm.graph_init(X,c,F,float(input1) , float(input2) , float(input3) , float(input4),int(input5),int(input6))
        update_plot(graph ,c ,X, func.PC(weight),perf ,C, func.SC(weight,c,distance_matrix, m,distance_centers) ,lamda,m)
    elif selected_algorithm == 'GAkfcm':
        graph , c , X, weight, distance_matrix, m, perf , C,distance_centers = gakfcm.graph_init(X,c,int(input5),int(input6), alfa)
        update_plot(graph ,c ,X , func.PC(weight),perf ,C,func.SC(weight,c,distance_matrix, m,distance_centers))
    elif selected_algorithm == 'GAfcm':
        graph , c , X, weight, distance_matrix, m,perf , C,distance_centers = gafcm.graph_init(X,c,int(input5),int(input6))
        update_plot(graph ,c ,X, func.PC(weight),perf,C, func.SC(weight,c,distance_matrix, m,distance_centers))
    elif selected_algorithm == 'GAkifcm':
        graph , c , X, weight, distance_matrix, m, lamda,perf , C,distance_centers = gakifcm.graph_init(X,c,F,float(input1) , float(input2) , float(input3) , float(input4),int(input5),int(input6), alfa)
        update_plot(graph ,c ,X, func.PC(weight),perf ,C, func.SC(weight,c,distance_matrix, m,distance_centers) ,lamda,m)



def update_plot(graphme ,c ,X ,PC, perf,C ,SC=0,lamda=None,m=None):
    global pattern_done
    selected_algorithm = algorithm_var.get()
    plot_ax.clear()
    
    plot_ax_perf.clear()
    selected_pattern = pattern_var.get()
    selected_dataset = dataset_var.get()
    if selected_dataset == "Iris":
        F = load_iris().target
    elif selected_dataset == "Wine":
        F = load_wine().target
    elif selected_dataset == "Breast Cancer":
        F = load_breast_cancer().target
    elif selected_dataset == "Seeds":
        F = pd.read_csv("seeds.csv")['V8'].values
    elif selected_dataset == "Dermatology":
        F = pd.read_csv("derm.csv")['class'].values
    
    grouped_results = func.group_array_values(F)
    grouped_values = func.group_array_values(graphme)
    similarity_percentage = func.calculate_similarity_percentage(grouped_results, grouped_values , X)


    
    x = int(selected_pattern.split("-")[0])
    y = int(selected_pattern.split("-")[1])
    for iter1 in range(c):
        plot_ax.scatter(X[graphme == iter1, x], X[graphme == iter1, y], marker='x')
    
    
    plot_ax.set_title(f'{algorithm_var.get()} Clustering')
    
    plot_canvas.draw()

    plot_ax_perf.plot(perf)
    if selected_algorithm in ["GAifcm","GAkifcm"]:
        plot_ax_perf.set_title(f'Objective functions Plot')
        ProfCenter_box.grid(row=0, column=0, pady=5,columnspan=1, sticky="ns")
        ProfCenter_box.delete(1.0, END)
        ProfCenter_box.insert(END, f'{algorithm_var.get()} Clustering Plot:\n  Similarity Percentage: {similarity_percentage}% \n  PC: {PC} \n  SC: {SC} \nObjective functions Plot: \n  Best Value: {round(perf[len(perf)-1],2)} \n  M: {round(m,2)}\n  Lamda {round(lamda,2)}')
    else:
        plot_ax_perf.set_title(f'Objective functions Plot')
        ProfCenter_box.grid(row=0, column=0, pady=5,columnspan=1, sticky="ns")
        ProfCenter_box.delete(1.0, END)
        ProfCenter_box.insert(END, f'{algorithm_var.get()} Clustering Plot:\n  Similarity Percentage: {similarity_percentage}% \n  PC: {PC} \n  SC: {SC} \nObjective functions Plot: \n  Best Value: {round(perf[len(perf)-1],2)}')
    plot_canvas_perf.draw()
    Center_box.delete(1.0, END)
    i = 1
    for cen in C:
        Center_box.insert(END, f"C{i}= {cen} \n")
        i+=1
    pattern_done = True



def toggle_inputs(event):
    selected_algorithm = algorithm_var.get()
    if selected_algorithm in ["GAifcm" , "GAkifcm"]:
        # Show the inputs
        m_start.grid(row=6, column=0, sticky=tk.W)
        input_one.grid(row=6, column=1, pady=2)
        m_end.grid(row=7, column=0, sticky=tk.W)
        input_two.grid(row=7, column=1,  pady=2)
        lamda_start.grid(row=8, column=0, sticky=tk.W)
        input_three.grid(row=8, column=1,  pady=2)
        lamda_end.grid(row=9, column=0, sticky=tk.W)
        input_four.grid(row=9, column=1,  pady=2)
        l_pop.grid(row=10, column=0, sticky=tk.W)
        input_pop.grid(row=10, column=1,  pady=2)
        l_gen.grid(row=11, column=0, sticky=tk.W)
        input_gen.grid(row=11, column=1,  pady=2)
        run_button.grid(row=13, column=0, columnspan=2, pady=10)
    elif selected_algorithm in ["GAkfcm" , "GAfcm"]:
        l_pop.grid_forget()
        l_gen.grid_forget()
        input_pop.grid_forget()
        input_gen.grid_forget()
        m_start.grid_forget()
        input_one.grid_forget()
        m_end.grid_forget()
        input_two.grid_forget()
        lamda_start.grid_forget()
        input_three.grid_forget()
        lamda_end.grid_forget()
        input_four.grid_forget()
        l_pop.grid(row=10, column=0, sticky=tk.W)
        input_pop.grid(row=10, column=1,  pady=2)
        l_gen.grid(row=11, column=0, sticky=tk.W)
        input_gen.grid(row=11, column=1,  pady=2)
        run_button.grid(row=13, column=0, columnspan=2, pady=10)
    else:
        # Hide the inputs
        l_pop.grid_forget()
        l_gen.grid_forget()
        input_pop.grid_forget()
        input_gen.grid_forget()
        m_start.grid_forget()
        input_one.grid_forget()
        m_end.grid_forget()
        input_two.grid_forget()
        lamda_start.grid_forget()
        input_three.grid_forget()
        lamda_end.grid_forget()
        input_four.grid_forget()
        run_button.grid(row=6, column=0, columnspan=2, pady=10)


def toggle_pattern(event):
    global pattern_done
    pattern_done = False
    selected_dataset = dataset_var.get()
    if selected_dataset == "Iris":
        values = ['']
        size = len(load_iris().data[0])
        for i in range(size):
            for j in range(i+1,size):
                values.append(f"{i}-{j}")
        values.pop(0)
        pattern_menu['values'] = values
        pattern_menu.current(0)
    
    elif selected_dataset == "Wine":
        values = ['']
        size = len(load_wine().data[0])
        for i in range(size):
            for j in range(i+1,size):
                values.append(f"{i}-{j}")
        values.pop(0)
        pattern_menu['values'] = values
        pattern_menu.current(0)
    
    elif selected_dataset == "Breast Cancer":
        values = ['']
        size = len(load_breast_cancer().data[0])
        for i in range(size):
            for j in range(i+1,size):
                values.append(f"{i}-{j}")
        values.pop(0)
        pattern_menu['values'] = values
        pattern_menu.current(0)
    
    elif selected_dataset == "Seeds":
        values = ['']
        size = len(pd.read_csv("seeds.csv").drop(['V8'], axis=1).values[0])
        for i in range(size):
            for j in range(i+1,size):
                values.append(f"{i}-{j}")
        values.pop(0)
        pattern_menu['values'] = values
        pattern_menu.current(0)
    
    elif selected_dataset == "Dermatology":
        values = ['']
        size = len(pd.read_csv("derm.csv").drop(['class'], axis=1).values[0])
        for i in range(size):
            for j in range(i+1,size):
                values.append(f"{i}-{j}")
        values.pop(0)
        pattern_menu['values'] = values
        pattern_menu.current(0)


def toggle_change(event):
    if pattern_done == True:
        update_plot(graph ,c ,X, func.PC(weight),perf ,C, func.SC(weight,c,distance_matrix, m,distance_centers) )

root = tk.Tk()
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Fcm Algorithms")

# Create a container frame for the inputs, combobox, and button
input_frame = ttk.Frame(root)
input_frame.grid(row=0, column=0, padx=40, pady=40)

dataset_var = tk.StringVar()
dataset_label = ttk.Label(input_frame, text="Select Dataset:")
dataset_label.grid(row=0, column=0, sticky=tk.W)
dataset_menu = ttk.Combobox(input_frame, textvariable=dataset_var, values=['Iris', 'Wine', 'Breast Cancer' , 'Seeds','Dermatology'])
dataset_menu.grid(row=1, column=0, pady=2)
dataset_menu.bind('<<ComboboxSelected>>', toggle_pattern)


pattern_var = tk.StringVar()
pattern_label = ttk.Label(input_frame, text="Select Pattern:")
pattern_label.grid(row=2, column=0, sticky=tk.W)
pattern_menu = ttk.Combobox(input_frame, textvariable=pattern_var)
pattern_menu.grid(row=3, column=0, pady=2)
pattern_menu.bind('<<ComboboxSelected>>', toggle_change)



algorithm_var = tk.StringVar()
algorithm_label = ttk.Label(input_frame, text="Select Algorithm:")
algorithm_label.grid(row=4, column=0, sticky=tk.W)
algorithm_menu = ttk.Combobox(input_frame, textvariable=algorithm_var, values=['Fcm', 'Kfcm', 'Ifcm', 'Kifcm', 'GAfcm', 'GAkfcm', 'GAifcm', 'GAkifcm'])
algorithm_menu.grid(row=5, column=0, pady=2)
algorithm_menu.bind('<<ComboboxSelected>>', toggle_inputs)

m_start = ttk.Label(input_frame, text="M Start:")
m_end = ttk.Label(input_frame, text="M End:")
lamda_start = ttk.Label(input_frame, text="Lamda Start:")
lamda_end = ttk.Label(input_frame, text="Lamda End:")
input_one = tk.Entry(input_frame)
input_two = tk.Entry(input_frame)
input_three = tk.Entry(input_frame)
input_four = tk.Entry(input_frame)
l_pop = ttk.Label(input_frame, text="Population:")
input_pop = tk.Entry(input_frame)

l_gen = ttk.Label(input_frame, text="Generations:")
input_gen = tk.Entry(input_frame)

run_button = ttk.Button(input_frame, text="Run Algorithm", command=run_algorithm)

graph_frame = ttk.Frame(root)
graph_frame.grid(row=0, column=1, pady=15)


Profermance_frame = ttk.Frame(root)
Profermance_frame.grid(row=0, column=3, padx=40, pady=40)
ProfCenter_box = Text(Profermance_frame, height = 10)



Center_box = Text(root, height = 10)
Center_box.grid(row=1, column=1, pady=5,columnspan=2, sticky="ew")

v=Scrollbar(root, command=Center_box.yview)
v.grid(row=1, column=0, sticky='nse')
Center_box['yscrollcommand'] = v.set


graph_frame_perf = ttk.Frame(root)
graph_frame_perf.grid(row=0, column=2, pady=5)

plot_figure = Figure(figsize=(4, 4), dpi=100)
plot_ax = plot_figure.add_subplot(111)
plot_ax.set_title('Algorithm Plot')
plot_canvas = FigureCanvasTkAgg(plot_figure, master=graph_frame)
plot_canvas.get_tk_widget().pack()

plot_figure_perf = Figure(figsize=(4, 4), dpi=100)
plot_ax_perf = plot_figure_perf.add_subplot(111)
plot_ax_perf.set_title('Objective functions Plot')

plot_canvas_perf = FigureCanvasTkAgg(plot_figure_perf, master=graph_frame_perf)
plot_canvas_perf.get_tk_widget().pack()

root.mainloop()