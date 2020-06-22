from utilities import get_data, plot_heatmap, plot_res
minIdx = 0
count = 0.
best = 0

for k in range(0,100,5):
    pred = []
    total = 0.
    for i in range(15):
        count = 0.
        for j in range(10):       
            minIdx = KNN(test_features[i*10+j], im_features, k)
            pred.append(minIdx)
            if minIdx == i:
                count += 1.
        total += count
    if total>best:
        y_pred = pred
    print(k, "total:", total/150.)
    
y_pred=np.array(y_pred)
y_true = list(test_y.astype(int))

plot_heatmap(y_true,y_pred,'./result/knn')
plot_res(y_true,y_pred,'./result/knn2')