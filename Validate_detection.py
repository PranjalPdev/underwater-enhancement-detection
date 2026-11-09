# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 11:25:26 2026

@author: pppan
"""



from ultralytics import YOLO

model = YOLO(r"C:\final\model\best.pt")
metrics = model.val(data=r"C:\internnn\URPC2020\data.yaml")

print("mAP50-95:", metrics.box.map)
print("mAP50:", metrics.box.map50)
print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)
