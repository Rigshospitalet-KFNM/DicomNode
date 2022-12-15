# Introduction

This document is focus around developing maintainable python applications. It contains tips and tricks how to ensure that your program remains reasonable. This is a programming term that means how well you are able to reason about the program so its structure and what different parts of the program is doing.

A very important note is that you can produce correct programs without following a single tip in this document, however when you need to work with them you will encounter additional difficulty.

## Simple programming types

### Baby steps - Following the style guide

A style guide is a number of self imposed restrictions, that when followed allow you to decern meaning from code. The Python standard library has a style guide in the python enhancement proposals or PEP for short: **<https://peps.python.org/pep-0008/>**, which describes how to write most code. Now style guides are something people fight over like religious wars, and if you use multiple libraries, these libraries might use different style guides, leaving your program looking a bit messy. Like not even the python standard library follow their own style guide.

I would recommend the following style guides:

* PascalCase for Classes - ThisIsPascalCase
* snake_case for general code - this_is_snake_case
* SCREAMING_SNAKE_CASE for constants - THIS_IS_SCREAMING_SNAKE_CASE

### Adding a doc string

A documentation string is a string which describe the functionality of a function or class. It should be placed after each function. Like so:

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

Now if you have a good editor with intelliSense (**<https://code.visualstudio.com/>** if you need one) they will display this docstring when you hover over the function. You can also document modules, by filling the first like of code in a file. Docstring are different as they actually gets stored in an object under the keyword `__doc__`. Mean you can accesoo....
```python
def function(arg_1: ArgType, kw_arg_1: KwArgType = kw_val) -> ReturnType:
  variable: VarType = ...
```

Now, some times you would like to use the flexibility of type system, and then you need the `typing` Library. It contains some helpful Classes:

* Any - This an explicit way of saying, I don't know the type of this object.
* Dict - For the build in `dict` class, this is a composite type, and is specified: `Dict[KeyType, ValueType]`
* List - For the build in `list` class, this is a composite type, and is specified: `List[ListType1, ListType2, ...]`
* Optional - Allows the variable to be either the type or `None`. it's specified as: `Optional[Type]`
* Type - In python3 types and classes
* Union - allows a variable to store multiple types, this is composite type and is specified: `Union[Type1, Type2, ...]` Now if you're a fancy and are using python3.10 or later you can specify this as `Type1 | Type2 | ...`

### Inheritance

This library heavily use inheritance, the idea is to 

## Advanced Documentations

This section will describe various design decisions of the library. Along with some VERY sharp corners of python. That probably will involve that python is a compiled and interpreted language. Now this is fairly well abstracted away in the language, however there's some few cases, where that actually matter. If you end in those you are about to have a bad time.

### Dynamically assigned functions

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
    self.method_4() # Raises TypeError, method 4 is given 1 argument but expects 0
```

While it's strongly recommend that you use method 1 for function declarations when you're able, you should notice that the call signatures are different of the call signature of the 3 functions.
Method 3 doesn't have the self parameter. Now if you remove the self parameter from method 1 or 2 you'll get a nice error, telling you that there is a mismatch between the number of arguments given and the call signature. This is because the file is compiled before it's interpreted. Hence when the compiler reached method 3 & 4 it determines its not in class definition so it's just a normal function. While for method 1 & 2 it's in a class definition and the functions have not been specified to be a static or a class method, hence it must be a instance method, and therefore require a self argument.
The tricky and perhaps weird part is when you assign functions to class, if you do it dynamically, then no it's assumed that's no self argument, while if you do it static, it does.

Okay why is this relevant? Well, the idea with this library is that you get the base classes and then overwrite various attribute with your own, and through that you end up with a functional pipeline. Now a goal is to minimize the shared code base between pipelines, or rather place that code in the library instead of pipeline source code. This means you cannot easily allow the library user to select a function from a catalog.
There's two solutions to this problem, either you add a self argument to method 4 and it means that you can't use method as a none instance method. The solution this library uses is not to class method 4 directly, but instead pass it first to a decorator, that removes the first argument.

Now you might think you're smart, by wrapping the function in the build-in `staticmethod`, but this does't work. Trust me I've tried.
