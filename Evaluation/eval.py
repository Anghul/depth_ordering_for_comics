#!/usr/bin/env python
# coding: utf-8

from PIL import Image
import torchvision.transforms as transforms
import os.path
import statistics

# evaluate the accuracy on the given image
def evalulate_image(prediction, gt_depth_ordering):
    with open(gt_depth_ordering) as file:
        lines = file.readlines()
    n = len(lines)
    goodl1 = 0
    alll1 = 0
    goodl2 = 0
    alll2 = 0
    for i1 in range(n):
        line1 = lines[i1].replace("\n","")
        l1, l2, x, y = line1.split(" ")
        l1, l2, x, y = int(l1), int(l2), int(x), int(y)
        for i2 in range(i1):
            line2 = lines[i2].replace("\n","")
            l1_, l2_, x_, y_ = line2.split(" ")
            l1_, l2_, x_, y_ = int(l1_), int(l2_), int(x_), int(y_)
            
            if l1<l1_:
                goodl1 += int(prediction[y][x] > prediction[y_][x_])
                alll1 += 1
            elif l1_<l1:
                goodl1 += int(prediction[y_][x_] > prediction[y][x])
                alll1 += 1
            else:
                if l2<l2_:
                    goodl2 += int(prediction[y][x] > prediction[y_][x_])
                    alll2 += 1
                elif l2_<l2:
                    goodl2 += int(prediction[y_][x_] > prediction[y][x])
                    alll2 += 1
    return goodl1, alll1, goodl2, alll2

# print the statistics based on an accuracy list
def print_accuracy(listacc):
    if len(listacc) == 0:
        print("no value")
    elif len(listacc) == 1:
        print("result", listacc[0])
    else:
        print("-mean()", statistics.mean(listacc))
        print("-quantiles()", statistics.quantiles(listacc))
        print("-stdev()", statistics.stdev(listacc))
        print("-len()", len(listacc))
        print("-len>0()", sum([x>0 for x in listacc]))
        print("-len>1()", sum([x>1 for x in listacc]))

# evaluate the model
def evaluate():
    # change model_dir depending on which model is evaluated
    model_dir = "./input/adabins"
    midas_dir = "./input/midas"
    gt_dir = "./input/ground_truth"

    with open("input/input.txt") as file:
        names = file.read().split("\n")
        
    import math

    inter_model = []
    intra_model = []
    inter_midas = []
    intra_midas = []
    inter_relat = []
    intra_relat = []
    inter_relat_log = []
    intra_relat_log = []


    for name in names[:-1]:   
        print(name)
        model_name = os.path.join(model_dir, name+".png")
        model = Image.open(model_name)
        midas_name = os.path.join(midas_dir, name+"-dpt_beit_large_512.png")
        midas = Image.open(midas_name)

        gt_name = os.path.join(gt_dir, name+".txt")

        model = transforms.ToTensor()(model)[0,:,:]
        midas = transforms.ToTensor()(midas)[0,:,:]

        goodl1, alll1, goodl2, alll2 = evalulate_image(model, gt_name)
        if alll1>0: 
            inter_model.append(goodl1/alll1) 
        if alll2>0: 
            intra_model.append(goodl2/alll2) 
        goodl1_, alll1_, goodl2_, alll2_ = evalulate_image(midas, gt_name)
        if alll1_>0: 
            inter_midas.append(goodl1_/alll1_) 
        if alll2_>0: 
            intra_midas.append(goodl2_/alll2_) 

        if goodl1_>0: 
            inter_relat.append(goodl1/goodl1_) 
        if goodl2_>0:
            intra_relat.append(goodl2/goodl2_) 

    for x in inter_relat:
        if x != 0:
            inter_relat_log.append(10*math.log10(x))
    for x in intra_relat:
        if x != 0:
            intra_relat_log.append(10*math.log10(x))

    print ("interobjects accuracy model")
    print_accuracy(inter_model)   
    print ("intraobjects accuracy model")  
    print_accuracy(intra_model)    
    print ("interobjects accuracy midas")
    print_accuracy(inter_midas)   
    print ("intraobjects accuracy midas")  
    print_accuracy(intra_midas)          
    print ("relative interobjects")
    print_accuracy(inter_relat)   
    print ("relative intraobjects")  
    print_accuracy(intra_relat)    
    print ("relative interobjects")
    print_accuracy(inter_relat_log)   
    print ("relative intraobjects")  
    print_accuracy(intra_relat_log)
                
if __name__ == "__main__":
    evaluate()