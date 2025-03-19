import numpy as np

class Dog:
    species = "Canis lupus"  
    # 类变量 在类内部直接定义，但不在任何方法（如__init__）内
    def __init__(self, name):
        self.name = name  
      # 实例变量 在类的构造函数（如__init__）或其他方法中通过self绑定

# 作用域：属于单个实例，不同实例的同名变量互不影响。
# 生命周期：与实例共存，实例被销毁时消失。

# 创建两个实例
dog1 = Dog("Buddy")
dog2 = Dog("Max")

print(dog1.species)  # 输出: Canis lupus
print(dog2.species)  # 输出: Canis lupus

# 修改类变量
Dog.species = "Canis lupus familiaris"
print(dog1.species)  # 输出: Canis lupus familiaris（所有实例同步更新）

class Person:
    def __init__(self, name, age):
        self.name = name  # 实例变量
        self.age = age    # 实例变量

# 创建两个实例
p1 = Person("Alice", 30)
p2 = Person("Bob", 25)

print(p1.name)  # 输出: Alice
print(p2.name)  # 输出: Bob（各自独立）

def calculate_sum(a, b):
    result = a + b  # 局部变量
    print(result)

calculate_sum(3, 5)  # 输出: 8
#print(result)        # 报错：result未定义（超出作用域）