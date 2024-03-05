# Building Software in Python

This document is focus around developing maintainable python applications. It
contains tips and tricks how to ensure that your program remains reasonable.
This is a programming term that means how well you are able to reason about the
program so its structure and what different parts of the program is doing.

This guide is a much shorter, version of **<https://peps.python.org/>**

A very important note is that you can produce correct programs without following
a single tip in this document, however when you need to work with them you will
encounter additional difficulty if you do not.

## Simple programming types

### Baby steps - Following the style guide

A style guide is a number of self imposed restrictions, that when followed
allow you to decern meaning from code. The Python standard library has a style
guide in the python enhancement proposals or PEP for short:
**<https://peps.python.org/pep-0008/>**, which describes how to write most code.
Now style guides are something people fight over like religious wars, and if you
use multiple libraries, these libraries might use different style guides,
leaving your program looking a bit messy. Like not even the python standard
library follow their own style guide.

I would recommend the following style guides:

* PascalCase for Classes - ThisIsPascalCase
* snake_case for general code - this_is_snake_case
* SCREAMING_SNAKE_CASE for constants - THIS_IS_SCREAMING_SNAKE_CASE

### Adding Documentation

The most basic form of documentation is a documentation string, which is a
string which describe the functionality of a function or class. It should be
 placed after each function. Like so:

```python
def function(arg_1, arg_2):
  """Description of function

  Args:
    arg_1 (type of arg_1) - Description of arg_1
    arg_2 (type of arg_2) - Description of arg_2
  Returns:
    (return type) - Description of returned element
  Raises:
    Exception - Description of error case

  Example:
    >>> ret = function(1, 2)
    3
  """
```

Now if you have a good editor with intelliSense
(**<https://code.visualstudio.com/>** if you need one) they will display this
docstring when you hover over the function. You can also document modules, by
filling the first like of code in a file. Docstring are different as they
actually gets stored in an object under the keyword `__doc__`. Mean you can
 access the documentation at run time if you really need to.

If you're unsure of what to put in a documentation, try and think about the
responsibility of the function and write that. You should not assume that a
docstring covers for magical coding tricks. You should add a comment explaining
what is going on. If you're using python this should manly happen when you hit a
 sharp corner of python. There're some examples of this in the advanced section.

Adding documentation is work, which is much like an investment. It requires
resources to get rolling, namely time. Not all documentation is going to be
worth that time, but it's impossible to determine which documentation is useful
and which isn't. A good rule of thumb is too add documentation to all public
methods and functions. However you can often receive better results writing a
tutorial or creating dataflow diagrams.

### Type hints

While python is dynamically typed. It allows a great freedom for the programmer,
and that isn't always a good thing. These freedoms allow a programmer to speed
though development, but this speed allows bugs to creep in, namely type error.
A Type error occurs when you assume an attribute have a method or property,
which is does not.

Python also have a PEP for this: https://peps.python.org/pep-0484/

Now to help with this, you can add type hints to your functions. A type hint is
a super duper pinky promise, that a variable is of a certain type. In other
words you can break a type hint, and your python program will still run no
problem (assuming no type error happens). Your IDE might yell at you thou.

To declare type hint:

```python
def function(arg_1: ArgType, kw_arg_1: KwArgType = kw_val) -> ReturnType:
  variable: VarType = ...
```

Some types are composite. IE they contain other objects of a certain type.
For instance a list of number. To specify this you need the `typing` Library.
You can see this down below:

* Any - This an explicit way of saying, I don't know the type of this object.
* Dict - For the build in `dict` class, this is a composite type, and is
specified: `Dict[KeyType, ValueType]` - *achoo Reader Monad*
* List - For the build in `list` class, this is a composite type, and is
specified: `List[ListType1, ListType2, ...]` - *achoo List monad*
* Optional - Allows the variable to be either the type or `None`. it's specified
as: `Optional[Type]` - *achoo Maybe Monad*
* Type - In python3 types and classes
* Union - allows a variable to store multiple types, this is composite type and
is specified: `Union[Type1, Type2, ...]` Now if you're a fancy and are using
python3.10 or later you can specify this as `Type1 | Type2 | ...`

While this seams easy and fine, you can often run into problems when dealing
with polymorphic functions. Consider this function.

```python
def add(x, y):
  return x + y
```

And add some type hints:

```python
def add(x: int, y: int) -> int:
  return x + y
```

Now somebody on your team wants to use this function with floating points. And
you get the union type.

```python
def add(x: Union[int, float], y: Union[int, float]) -> Union[int, float]:
  return x + y
```

Or you can all the way and create type variable

```python
T = TypeVar('T', int, float, complex)

def func(var_1: T, var_2: T) -> T:
  return var_1 + var_2
```

But now you have to deal with the possibility of float returned in your code. A
possible solution is by overloading the method, but that's not supported by the
standard library. If a method starts to become too difficult to deal with type
wise, consider separating it into multiple functions.

Even if you franticly type hint EVERYTHING, the libraries that you use might not
be type hinted, leading you what is generally known as a bad time, so don't
sweet to hard about it.

### Inheritance

Python is an object oriented language, which means inheritance is a big part of
the language. Inheritance allows you to inherit functionality from super classes
ie:

```python
class SuperClass:
  def super_method(self):
    print("super method")

class SubClass(SuperClass):
  pass

>>> s = SubClass()
>>> s.super_method()
super method
```

Now say you some different functionality you can overwrite the function with
your own implementation:

```python
class SuperClass:
  def super_method(self):
    print("super method")

class SubClass(SuperClass):
  def super_method(self):
    print("sub method")

>>> s = SubClass()
>>> s.super_method()
sub method
```

This library utilized inheritance to provide some "templates" for user of the
library to fill out.
This also deals with problem of users with very specific requirements, by
allowing them to overwrite the part they want changed. So if you want to change
something you can probably overwrite it and pass it in as an option.

Note that it's generally frowned upon to create long inheritance chains.

#### Naming Conventions in python

In most object oriented languages, attributes and methods are encapsulated with
public and private access modifiers. In python everything is public, however by
convention methods and attributes starting with a `_` is considered private,
however there's nothing stopping you from overwriting this method.

If there's two underscores `__` then the method or attribute is subject to name
mangling.

Consider this example to see the difference:

```python

>>>class A:
...  __foo = "bar"

...  def print_foo(self)
...    print(self.__foo)

>>>class B(A):
...  __foo = "baz"

>>>b = B()
>>>b.print_foo()
bar

>>>class C:
...  _foo = "bar"

...  def print_foo(self)
...    print(self._foo)

>>>class D(C):
...  _foo = "baz"

>>>d = D()
>>>d.print_foo()
baz
```

## Advanced Documentations

This section will describe various design decisions of the library. Along with
some VERY sharp corners of python. Most of these are just here because I have
struggled with them for a couple of hours at one point or another and this is my
way of getting these frustrations out.

### Dynamically assigned functions

Python is a compiled and interpreted language. Now this is fairly well
abstracted away in the language, however there's some few cases, where that
actually matter. This showcases this:

There are a few different ways that you assign functions to a class.

```python
def method_4():
  pass

def method_3():
  pass

class A:
  def method_1(self):
    pass

  method_2 = lambda self: None

  method_4 = method_4
  def __init__(self)
    self.method_3 = method_3
    self.method_1() # ok
    self.method_2() # ok
    self.method_3() # ok
    # Raises TypeError, method 4 is given 1 argument but expects 0
    self.method_4()
```

While it's strongly recommend that you use method 1 for function declarations
when you're able, you should notice that the call signatures are different of
the call signature of the 3 functions.
Method 3 doesn't have the self parameter. Now if you remove the self parameter
from method 1 or 2 you'll get a nice error, telling you that there is a mismatch
between the number of arguments given and the call signature. This is because
the file is compiled before it's interpreted. Hence when the compiler reached
method 3 & 4 it determines its not in class definition so it's just a normal
function. While for method 1 & 2 it's in a class definition and the functions
 have not been specified to be a static or a class method, hence it must be a
 instance method, and therefore require a self argument.
The tricky and perhaps weird part is when you assign functions to class, if you
do it dynamically, then no it's assumed that's no self argument, while if you do
it static, it does.

Okay why is this relevant? Well, the idea with this library is that you get the
base classes and then overwrite various attribute with your own. It means you
cannot use a function as an attribute.

You can try and use the `staticmethod` decorator to try and fix this.

The pretty solution is just to create a callable object as a wrapper I.E. The
Grinders
