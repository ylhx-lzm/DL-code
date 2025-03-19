from typing import List
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dict = {}
        
        for i in range(len(nums)):
            if target - nums[i] not in dict:
                dict[nums[i]] = i
            else:
                return [dict[target-nums[i]], i]
solution = Solution()
ts = solution.twoSum([1,2,3], 5)
print(ts)
