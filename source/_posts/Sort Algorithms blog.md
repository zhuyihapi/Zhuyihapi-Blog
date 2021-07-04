---
title: Sort Algorithm
date: 2021-07-04
tag: Data structure and algorithm
---



# Sort Algorithm

## Contents

[toc]



## 1 [Quick Sort](http://data.biancheng.net/view/117.html)

### 1.1 Basic idea of the algorithm

通过一趟排序将要排序的数据分割成独立的两部分，其中一部分的所有数据比另一部分的所有数据要小，再按这种方法对这两部分数据分别进行快速排序，整个排序过程可以递归进行，使整个数据变成有序序列。

### 1.2 Principal of the algorithm

<!-- unfinished -->

### 1.3 Pseudo code

```pseudocode
quickSort(A, low, high)
  if low < high
  	i = low
  	j = high
  	pivot = A[low]
    while i < j do
    begin
    	while i < j and A[j] > pivot do
    	begin
    		j = j-1
    	end
    	if i < j then
    		A[i] = A[j]
    		i = i+1
    	while i < j and A[i] < pivot do
    	begin
    		i = i+1
    	end
    	if i < j then
    		A[j] = A[i]
    		j = j-1
    end
    A[i] = pivot 
    quicksort(A, low, i - 1)
    quicksort(A, i + 1, high)
```



### 1.4 Java implementation (others)

```java
public class QuickSort {
  private int[] array;
  //constructor
  public QuickSort(int[] array) {
    this.array = array;
  }
  //implement recursion
  public void sort() {
    quickSort(array, 0, array.length - 1);
  }
  //print all the elements in an array
  public void print() {
    for (int i = 0; i < array.length; i++) {
      System.out.println(array[i]);
    }
  }

  private void quickSort(int[] src, int begin, int end) {
    //首先判断输入的数组是否满足排序的条件（例如只剩下一个元素就不用排序了）
    if (begin < end) {
      //选择一个“基准”值，保存在变量pivot中，数组中产生一个“空位”
      int pivot = src[begin];
      int i = begin;
      int j = end;
      //注意任何操作之后都需要判断i是否小于j！
      while (i < j) { 
        //如果j指针指向的数大于pivot，那么不用交换，将j指针向前移动一位
        while (i < j && src[j] > pivot) { 
          j--;
        }
        //否则，将j指针指向位置的值赋给i指针指向的空位
        if (i < j) {
          src[i] = src[j];
          i++;
        }
        //对i指针进行相同的操作
        while (i < j && src[i] < pivot) {
          i++;
        }
        if (i < j) {
          src[j] = src[i];
          j--;
        }
      }
      //最终i,j指针指向同一个位置，并且这个位置必然是空位，将保存在pivot中的值放入该位置
      src[i] = pivot; 
      //对该位置两端进行递归的快速排序，注意作为上次“基准”的位置不再参与之后的排序
      quickSort(src, begin, i - 1);
      quickSort(src, i + 1, end);
    }
  }
}
```



### 1.5 Algorithm analysis

#### Efficiency

- $O(n \log_2 n)$ on average and best
- $O(n^2)$ worst case

#### Space usage

- $O(1)$



## 2 Merge Sort

### Pseudo code

```pseudocode
//input: array: A[1..n], index of array: left, right
//output: ascending array: A[left..right]

MergeSort(A,left,right)
if left >= right then
	return A[left..right]
end
mid <- ⌊(left+right)/2⌋
MergeSort(A,left,mid)
MergeSort(A,mid+1,right)
Merge(A,left,mid,right)
return A[left..right]

Merge(A,left,mid,right)
//initialize
A′[left..right] <- A[left..right]
i <- left, j <- mid, k <- 0
while i <= mid and j <= right do
begin
	if A′[i] <= A′[j] then
		A[left+k] <- A′[i]
		k <- k+1, i <- i+1
	end
	else
		A[left+k] <- A′[j]
		k <- k+1, j <- j+1
	end
end
if i <=mid then
	A[left+k..right] <- A′[i..mid]
end
else
	A[left+k..right] <- A′[j..mid]
end
return A[left..right]
```

### Java implementation (other)

```java
import java.util.Arrays;

public class MergeSort {
    public static void merge(int[] a, int low, int mid, int high) {
        int[] temp = new int[high - low + 1]; // 初始化一个临时数组用于暂时保存数组融合的结果
        int left = low; // 左指针
        int right = mid + 1; // 右指针
        int k = 0;
        // 比较两个待融合的数组中指针指向的位置，将值小的先移入temp
        while (left <= mid && right <= high) {
            if (a[left] < a[right]) {
                temp[k++] = a[left++];
            } else {
                temp[k++] = a[right++];
            }
        }
        // 把左边剩余的数移入数组
        while (left <= mid) {
            temp[k++] = a[left++];
        }
        // 把右边边剩余的数移入数组
        while (right <= high) {
            temp[k++] = a[right++];
        }
        // 把新数组中的数覆盖nums数组
        for (int i = 0; i < temp.length; i++) {
            a[low + i] = temp[i];
        }
    }

    public static void mergeSort(int[] a, int low, int high) {
        int mid = (low + high) / 2; // 这里需要向下取整吗？（不需要）
        // 当出现mid=low
        if (low < high) {
            // 左边
            mergeSort(a, low, mid);
            // 右边
            mergeSort(a, mid + 1, high);
            // 左右归并
            merge(a, low, mid, high);
            System.out.println(Arrays.toString(a));
        }

    }

    public static void main(String[] args) {
        int a[] = {51, 46, 20, 18, 65, 97, 82, 30, 77, 50};
        mergeSort(a, 0, a.length - 1);
        System.out.println("排序结果：" + Arrays.toString(a));
    }
}
```

### Algorithm analysis

#### Efficiency

- $O(n \log_2 n)$ on average
- The same as worst case and best case

#### Space usage

- $O(n)$



## 3 Insertion Sort

### Pseudo code

```pseudocode
//a is the array which stored the elements
//n is the number of elements in a[]

for i = 1 to n-1 do
begin
	pivot = a[i]
	position = 0
	while a[position]<pivot and position<i do
	begin
		shift a[position],...,a[i-1] to the right
		a[position] = pivot
	end
end
```

### Java implementation (others)

```java
public class insert_sort {
  public static void main(String[] args) {
    int[] unsorted = {1,-1,0,-2,3}; //34,8,64,51,32,21
    int[] sorted = new int[unsorted.length]; //default element is 0
    //insert sort, ascending order
    for (int i = 0; i < unsorted.length; i++) {
      for (int j = 0; j < sorted.length; j++) {
        if (unsorted[i] <= sorted[j]){ //<= or <
          //位置之后的数全部后移一位
          for (int k = sorted.length-1; k > j; k--) { //必须倒序！
            sorted[k] = sorted[k-1];
          }
          sorted[j] = unsorted[i];
          break;
        }else if (j == sorted.length-1){ //本次遍历没有发生交换，意味着待插入的新值大于已插入的所有值，包括默认值0
          sorted[i] = unsorted[i];
          break;
        }
      }
      //测试每一次插入新值后的sorted数组
      /*System.out.println("---"+i+"---");
            for (int element :
                    sorted) {
                System.out.println(element);
            }*/
    }

    //输出结果
    //System.out.println("=======final=======");
    for (int element :
         sorted) {
      System.out.println(element);
    }
  }
}

```



## 4 Bubble Sort

### Java implementation

```java
//Ascending order
public class BubbleSort {
    public static void main(String[] args) {
        double[] original = {2, 1, -3, 1};

        for (double a : bubbleSort(original)) {
            System.out.println(a);
        }
    }

    public static double[] bubbleSort(double[] array) {

        for (int i = 0; i < array.length; i++) {
            boolean swapped = false;
            for (int j = array.length - 1; j > i; j--) {
                if (array[j - 1] > array[j]) {
                    double temp = array[j];
                    array[j] = array[j - 1];
                    array[j - 1] = temp;
                    swapped = true;
                }
            }
            if (!swapped) {
                break;
            }
        }
        
        return array;
    }
}
```



### Algorithm analysis

#### Efficiency

- Worst case: $O(n^2)$
- Average case: $O(n^2)$

#### Requirements on Data

- Needs random-access data, but can modify for files
- Needs compare and swap

#### Space usage

- in-space



## 5 Selection Sort

### Java implementation

```java
//Ascending order
public class SelectionSort {
    public static void main(String[] args) {
        double[] original = {2, 1, -3, 1};

        for (double a : selectionSort(original)) {
            System.out.println(a);
        }
    }

    public static double[] selectionSort(double[] array) {

        for (int i = 0; i < array.length - 1; i++) {
            int minimum = i;
            for (int j = i + 1; j < array.length; j++) {
                if (array[j] < array[minimum]) {
                    minimum = j;
                }
            }
            if (minimum != i) {
                double temp = array[i];
                array[i] = array[minimum];
                array[minimum] = temp;
            }
        }
        return array;
    }
}
```

### Algorithm analysis

#### Efficiency

- Worst case: $O(n^2)$
- Average case: $O(n^2)$

#### Requirements on Data

- Needs random-access data, but can modify for files
- Needs compare and swap

#### Space usage

- in-space ($O(1)$)



## Other

### [Array Sorting Algorithms Analysis](https://itimetraveler.github.io/2017/07/18/%E5%85%AB%E5%A4%A7%E6%8E%92%E5%BA%8F%E7%AE%97%E6%B3%95%E6%80%BB%E7%BB%93%E4%B8%8Ejava%E5%AE%9E%E7%8E%B0/)

<img src="https://raw.githubusercontent.com/zhuyihapi/picture/main/20210602145650.png" alt="img" style="zoom: 80%;" />

### [Time consuming sorting algorithm](https://github.com/iTimeTraveler/SortAlgorithms/blob/master/BENCH_RESULT.md)

​	

### Binary Search

#### Pseudo code

```pseudocode
RecurBinarySearch(A,first,last,X)
begin
	if(first>last) then
		return false
	mid = ⌈(first+last)/2⌉
	if(X == A[mid]) then
		return true
	if(X<A[mid]) then
		return RecurBinarySearch(A,first,mid-1,X)
	else
		return RecurBinarySearch(A,mid+1,last,X)
end
```

