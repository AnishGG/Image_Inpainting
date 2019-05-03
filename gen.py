import random
N = 5
n = 5
m = 15

print(N, n, m)

# Training X
for i in range(N*n):
    print(random.random(), end=" ") 
print()

# Training Y
for i in range(N*n):
    print(random.random(), end=" ")
print()

# Testing X
for i in range(N*n):
    print(random.random(), end=" ")
print()
