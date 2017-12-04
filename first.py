#from sympy import *
def f1():# while loop
	print('----------')
	c=-20
	dc=5
	while c<=40:
		f=(9.0/5)*c+32
		print(c,f)
		c=c+dc
	print('----------')

def f2(n): # for loop
	print('----------')
	for c in range(0,n):
		f=(9.0/5)*c+32
		print(c,f)
	print('----------')

def f3(x):#for loop 
	print('----------')
	for c in x:
		f=(9.0/5)*c+32
		print(c,f)
	print('-----------')


f1()
f2(10)
x=[1,5,10]
f3(x)
