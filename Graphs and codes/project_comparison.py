import matplotlib.pyplot as plt

YOLO_TIME=[1.762765645980835, 1.5139868259429932, 1.495448112487793, 1.5169830322265625, 1.5680692195892334, 1.54152250289917, 1.5715725421905518, 1.549034833908081, 1.5440266132354736, 1.5971183776855469, 1.5465309619903564, 1.5330092906951904, 1.5199899673461914, 1.5705790519714355, 1.5555458068847656, 1.6146447658538818, 1.5450310707092285, 1.602630853652954, 1.5129787921905518, 1.5144810676574707, 1.4924464225769043, 1.4949488639831543, 1.4899399280548096, 1.4964518547058105, 1.5124754905700684, 1.496455430984497, 1.494947910308838, 1.4704086780548096, 1.494448184967041, 1.4799244403839111, 1.510472059249878, 1.5014586448669434, 1.4939465522766113, 1.4869346618652344, 1.4789214134216309, 1.4924440383911133, 1.4714083671569824, 1.482928991317749, 1.483931303024292, 1.4769182205200195, 1.5650649070739746, 1.559053897857666, 1.5159804821014404, 1.624157190322876, 1.7839195728302002, 1.7388453483581543, 1.4799220561981201, 1.479924201965332, 1.5074715614318848, 1.5104734897613525, 1.5004565715789795, 1.491445541381836, 1.502962589263916, 1.530503273010254, 1.5450303554534912, 1.507969617843628, 1.4859325885772705, 1.4949448108673096, 1.4964499473571777, 1.7017858028411865, 1.7653915882110596, 1.8104636669158936, 1.7849245071411133, 1.6992802619934082, 1.710801601409912, 1.652703046798706, 1.5159835815429688, 1.5380175113677979, 1.5234949588775635, 1.5109717845916748, 1.5219926834106445, 1.5234954357147217, 1.5114760398864746, 1.5024619102478027, 1.5164804458618164, 1.5074687004089355, 1.4674029350280762, 1.5069680213928223, 1.5049655437469482, 1.5174849033355713, 1.4949471950531006, 1.5029609203338623, 1.4794225692749023, 1.5114762783050537, 1.4784226417541504, 1.5109734535217285, 1.5049619674682617, 1.4934461116790771, 1.492443323135376, 1.477919340133667, 1.477414608001709, 1.47491455078125, 1.4759173393249512]

HOG_TIME=[0.16977405548095703, 0.20133042335510254, 0.19481897354125977, 0.1712801456451416, 0.18580293655395508, 0.1737833023071289, 0.17528629302978516, 0.17478537559509277, 0.19181418418884277, 0.20383358001708984, 0.17829132080078125, 0.19982624053955078, 0.17728972434997559, 0.1878070831298828, 0.17829132080078125, 0.17278242111206055, 0.18079495429992676, 0.16977667808532715, 0.193817138671875, 0.17378497123718262, 0.18730688095092773, 0.17178106307983398, 0.17027854919433594, 0.1737833023071289, 0.17829036712646484, 0.16977643966674805, 0.1742851734161377, 0.18029379844665527, 0.18931007385253906, 0.17027854919433594, 0.17228007316589355, 0.17228174209594727, 0.17628717422485352, 0.1878063678741455, 0.17077946662902832, 0.17178082466125488, 0.17228150367736816, 0.1737804412841797, 0.16877532005310059, 0.1737837791442871, 0.18079519271850586, 0.18079495429992676, 0.19231510162353516, 0.16977596282958984, 0.18981099128723145, 0.17778921127319336, 0.18079543113708496, 0.19231295585632324, 0.16927695274353027, 0.17829084396362305, 0.16927599906921387, 0.20884060859680176, 0.16877412796020508, 0.1853020191192627, 0.196319580078125, 0.16927623748779297, 0.1792924404144287, 0.17278456687927246, 0.16777396202087402, 0.19782352447509766, 0.16476917266845703, 0.19181323051452637, 0.17528629302978516, 0.17228150367736816, 0.16777396202087402, 0.18980813026428223, 0.1797928810119629, 0.17578673362731934, 0.17278313636779785, 0.16977739334106445, 0.19832348823547363, 0.19131207466125488, 0.1742854118347168, 0.17928814888000488, 0.18730616569519043, 0.1797950267791748, 0.18380093574523926, 0.19034171104431152, 0.19982671737670898, 0.17628908157348633, 0.185302734375, 0.17278289794921875, 0.17178082466125488, 0.17728900909423828, 0.1797940731048584, 0.17428374290466309, 0.18580365180969238, 0.17829179763793945, 0.17028117179870605, 0.1737840175628662, 0.16777372360229492, 0.17228078842163086, 0.1772899627685547]

FRAME=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92]
sum_yolo=0
sum_hog=0
avg_yolo=0
avg_hog=0
# line 1 points
# plotting the line 1 points  
plt.plot(FRAME,HOG_TIME, label = "HOG PERFORMANCE")
for i in YOLO_TIME:
    sum_yolo=sum_yolo+i
avg_yolo=sum_yolo/92
for j in HOG_TIME:
    sum_hog=sum_hog+j
avg_hog=sum_hog/92

print(avg_hog,"-",avg_yolo)
  
# line 2 points 

# plotting the line 2 points  
plt.plot(FRAME,YOLO_TIME, label = "YOLO PERFORMANCE") 
  
# naming the x axis 
plt.xlabel('FRAMES') 
# naming the y axis 
plt.ylabel('TIME ELAPSED') 
# giving a title to my graph 
plt.title('HOG VS YOLO COMPARISON') 
  
# show a legend on the plot 
plt.legend() 
  
# function to show the plot 
plt.show() 
