# dit is een document waarin ik leer wat een python class is en doet en kan en hoe het handig gebruikt kan worden

class MyClass:
	"""a simple example class"""
	i = 12345
	
	def f(self):
		return "hello world"
		
x = MyClass()

print x.i
print x.f() # without the hooks it would be a function OBJECT! Now it's calling the function
# is equivalent to: Myclass.f(x)!!!! whoa

# deze klasse creert nu een leeg object. Als je wil dat een member van de class al geinstantieerd wordt, dan kun je een init functie doen:


class AnotherClass:
	"""a class which initiates its members"""
	def __init__(self):
		self.data = []
		
y = AnotherClass()

print y.data

class Proposition:
	"""a class to which you can pass arguments which will become part of the object's initialisation"""
	def __init__(self,l,r):
		self.left = l
		self.right = r

p = Proposition("LeftHand", "RightHand")
print p.left, p.right

# dit is een instance object
x.counter = 1
while x.counter < 10:
    x.counter = x.counter * 2
print x.counter

del x.counter

	