# coding=utf-8
def GetLeastNumbers_Solution( tinput, k):
    # write code here
    def quick_sort(lst):
        if not lst:
            return []
        pivot = lst[0]
        left = quick_sort([x for x in lst[1:] if x < pivot])
        right = quick_sort([x for x in lst[1:] if x >= pivot])
        print(left + [pivot] + right)
        return left + [pivot] + right
    if tinput == [] or k > len(tinput):
        return []
    tinput = quick_sort(tinput)
    return tinput[: k]
a=[3,4,6,2,7]
print (GetLeastNumbers_Solution(a,3))
