def merge(left,right):
    left_index,right_index = 0,0
    result=[]
    while left_index<len(left) and right_index<len(right):
        if left[left_index]<right[right_index]:
            result.append(left[left_index])
            left_index+=1
        else:
            result.append(right[right_index])
            right_index+=1
    result +=left[left_index:]
    result +=right[right_index:]
    return result


def merge_sort(a):
    if(len(a)<=1):
        return a
    mid = len(a)//2
    left = merge_sort(a[:mid])
    right = merge_sort(a[mid:])
    return merge(left,right)

def swap(x,y):
    temp = x
    x=y
    y= temp
    return x,y

def partition(a,lo,hi):
    pivot = a[lo]
    leftmark = lo+1
    rightmark = hi
    done = False
    while not done:
        while a[leftmark]<pivot:
            leftmark+=1
        while a[rightmark]>pivot:
            rightmark-=1
        if(leftmark>=rightmark):
            done = True
        else:
            a[leftmark],a[rightmark]=swap(a[leftmark],a[rightmark])
    a[lo],a[rightmark]=swap(a[lo],a[rightmark])
    return rightmark

def quick_sort(a,lo,hi):
    if(lo<hi):
        mid = lo+(hi-lo)//2
        p=partition(a,lo,hi)
        quick_sort(a,lo,mid-1)
        quick_sort(a,mid+1,mid+1)
        return a

a=[1,4,3,9,6,7]
result = merge_sort(a)
quick_sort(a,0,len(a)-1)
for i in result:
    print(i)
