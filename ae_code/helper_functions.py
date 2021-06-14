'''
Helper Functions to for general tasks.

Functions:

    num_elements(list) -> int
    
'''


def num_elements(list_of_elements):
    '''Take in a list, return number of elements in nested list'''
    count = 0
    for elem in list_of_elements:
        if type(elem) == list:
            count += num_elements(elem)
        else:
            count += 1
    return count

