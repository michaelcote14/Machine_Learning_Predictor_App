import random


for i in range(50):
    SampledNumber = random.sample(range(3), 1) #((random range of numbers)), how many numbers in list)
    TestList = ['apple', 'banana', 'orange']
    SampledList = random.sample(TestList, SampledNumber[0]+1)

    print(SampledNumber)
    print(SampledList)

# sampled_list = random.sample(TestList, random.sample(range(5), 1))
# print(sampled_list)

CombinationCount = 0
# for k in TestList:
#     print("k:", k)
#     for j in TestList:
#         print("j:", j)
#         if j != k:
#             print("working")
#             print("k, j:", k,j)
#         CombinationCount = CombinationCount + 1
#         print("Combination Count:",CombinationCount)
