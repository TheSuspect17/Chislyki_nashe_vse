import csv
from matrix import type_conversion



with open("d://input_inter.csv", encoding='utf-8') as r_file:
    file_reader = csv.reader(r_file, delimiter=";")
    x = []
    y = []
    for row in file_reader:
        x.append(type_conversion(row[0]))
        y.append(type_conversion(row[1]))


#type_conversion можно заменить на float 
#Название csv-шника d://input_inter.csv . Меняется при желании 
