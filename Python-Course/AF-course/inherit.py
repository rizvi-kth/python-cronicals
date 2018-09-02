x = 10
class Employee:
    # Class level variables - like static variables in other languages
    age = 12

    def __init__(self, name):
        self.name = name
        self.age = 24

    def __str__(self):
        return "<Employee: {}>".format(self.name)

    def getAge(self):
        return self.age
        #Instance methods see only the instance level methods.

    @classmethod
    def getAge1(klass):
        return klass.age
        #Classmothods see all the class level variables

    @staticmethod
    def getAge2():
        return x
        #Static methods dont see any variables in the containing class 


class EmployeeInside(Employee):
    def __str__(self):
        return super().__str__() + "<Inside>"


el = Employee("Rizvi")
print(el.getAge())
print(el.getAge1())
print(el.getAge2())


e2 = EmployeeInside("Boris")
print(e2)
