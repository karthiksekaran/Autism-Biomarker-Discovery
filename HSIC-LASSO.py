from pyHSICLasso import HSICLasso
hsic_lasso = HSICLasso()
hsic_lasso.input("SNR-26415.csv")
print(hsic_lasso.classification(100))
hsic_lasso.get_features()
l = []
l.append(hsic_lasso.get_features())
print(hsic_lasso.get_features())
print(len(l))
temp = 0
hsic_lasso.dump()
for i in range(0,len(l)):
    print(l[i])
    temp = temp + 1
print(temp)