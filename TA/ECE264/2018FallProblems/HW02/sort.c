// ***
// *** You MUST modify this file, only the ssort function
// ***

#include "sort.h"
#include <stdio.h>
#include <stdbool.h>

static bool checkOrder(int * arr, int size)
// a static function is visible only in this file
// This function returns true if the array elements are
// in the ascending order.
// false, otherwise
{
  int ind;
  for (ind = 0; ind < (size - 1); ind ++)
    {
      if (arr[ind] > arr[ind + 1])
    {
      return false;
    }
    }
  return true;
}
void printArray1(int * arr, int size)
{
  int ind;
  for (ind = 0; ind < size; ind ++)
    {
      fprintf(stdout, "Step! %d\n", arr[ind]);
    }
}


#ifdef TEST_SORT
void ssort(int * arr, int size)
{
  // This function has two levels of for
  // The first level specifies which element in the array
  // The second level finds the smallest element among the unsorted
  // elements

  // After finding the smallest element among the unsorted elements,
  // swap it with the element of the index from the first level
  
  // call checkOrder to see whether this function correctly sorts the
  // elements

  
  int tempSmallestIndex=0,tempNumber=0,i=0,j=0;
  
  for (i = 0; i < size - 1; i++)
    {
      printf("i = %d\n", i);
      tempSmallestIndex = i;
      for (j = i + 1; j < size; j++)
    {
      printf("j = %d\n",j );
      if (arr[tempSmallestIndex] > arr[j])
      {
        tempSmallestIndex = j;
      }
    }
    printf("Should change: %d\n", arr[tempSmallestIndex]);
      tempNumber = arr[i];
      arr[i] = arr[tempSmallestIndex];
      arr[tempSmallestIndex] = tempNumber;
      //printArray1(arr, size);
    }
  
  if (checkOrder(arr, size) == false)
    {
      fprintf(stderr, "checkOrder returns false\n");
    }
}
#endif