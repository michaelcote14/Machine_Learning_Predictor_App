import random

#ToDo convert numbers in a list to integer format
sampled_number = random.sample(range(4), 1) #((random range of numbers)), how many numbers in list)
#sampled_number2 = int(sampled_number)
print("sampled_number:", int(sampled_number[0]))
print("type:", type(sampled_number[0]))


aList = [20, 40, 80, 100, 120]
sampled_list = random.sample(aList, sampled_number[0]+1)
print("sampled_list:", sampled_list)
print("length:", len(sampled_list))
