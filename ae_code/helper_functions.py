def num_elements(list_of_elements):
    '''Get number of elements in nested list'''
    count = 0
    for elem in list_of_elements:
        if type(elem) == list:
            count += num_elements(elem)
        else:
            count += 1
    return count

