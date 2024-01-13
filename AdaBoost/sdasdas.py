
import numpy as np

print( 1* [True, False, True],)


x = np.array([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]])

                
#print(-1*x[:,2] == 3)

stumps = [1,2,3,4,5,6,7,8,9]

s =sum( [1 if stump==2 else 0 for stump in stumps  ])

print(np.zeros(5))