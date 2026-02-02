#numpy basics for ML
import numpy as np
print("================== Numpy Basics ================ ")

#1D array
arr=np.array([10,20,30,40,50])
print("Array:",arr)
print("Type of array:",type(arr))

print("\n=== basic statistics on 1D array ===")
print("Mean:",arr.mean())
print("Sum:",arr.sum())
print("Max:",arr.max())
print("Min:",arr.min())

print("\n====mathematical operations on 1D array====")
print("Addition:",arr+10)
print("Subtraction:",arr-5)
print("Multiplication:",arr*2)
print("Division:",arr/2)
print("Square:",arr**2)
print("Square Root:",np.sqrt(arr))

#2D array
print("\n=== 2D Array (dataset representation) ===")
data=np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
    ])
print("dataset:\n",data)

print("\nfirst row:",data[0])
print("second column:",data[:,1])
print("element at (1,1):",data[1,1])

print("\n=== feature and label separation ===")
x=data[:,1]  #features (study hours)
y=data[:,0]  #labels (marks)
print("Features (x):",x)
print("Labels (y):",y)

print("\n=== done with numpy basics ===")
