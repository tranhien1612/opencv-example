# python

## Installation Anaconda

Install anaconda for windows
```
    https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Windows-x86_64.exe
```

Create an environment:
```
    conda create -n <name_environment> python=3.10
```

Remove environment:
```
    conda deactivate
    conda remove --name <name_environment> --all
```

Show list environments:
```
    conda info --envs
```

Activate environment:
```
    conda activate <name_environment>
```

## Getting Started with Python Programming

Check python version:
```
    python --version
```
### Input/Output
```
    age_input = input("Enter your age: ")
    age = int(age_input)
    print("age", age)
```

### Comments
```
    # This is a comment
    print("Hello, World!")  # This is another comment
```

### Variables
```
    x = 10  # Integer
    y = 3.14  # Float
    name = "John"  # String
```

### Data Type
- Integers: Whole numbers without decimals.
- Floats: Numbers with decimals.
- Strings: Text enclosed in single or double quotes.
- Lists: Ordered collections of items.
- Tuples: Immutable collections of items.
- Dictionaries: Key-value pairs.

```
    x = 10  # Integer
    y = 3.14  # Float
    name = "John"  # String
    list = ["hello", "world"]
    tup = ('hello', 'world')
    dict = {"name": "Suraj", "age": 24}
    print("Type of x: ", type(x))
```

### Control Flow
If-else statement:
```
    x = 3
    if x == 4:
        print("Yes")
    else:
        print("No")
    
    letter = "A"
    if letter == "B":
        print("letter is B")

    elif letter == "C":
        print("letter is C")

    elif letter == "A":
        print("letter is A")

    else:
        print("letter isn't A, B or C")
```

For Loops:
```
    for i in range(10):
        if i == 5:
            break  # Exit the loop when i is 5
        print(i)
```

While Loops:
```
    while expression:
        statement(s)
```

### Functions
```
    def greet(name):
        print(f"Hello, {name}!")
    greet("world")  # Output: Hello, world!
```

### Operators
```
    + : Addition: adds two operands
    - : Subtraction: subtracts two operands
    * : Multiplication: multiplies two operands
    / : Division (float): divides the first operand by the second
    % : Modulus: returns the remainder when the first operand is divided by the second
    **: Power: Returns first raised to power second

    >  : Greater than: True if the left operand is greater than the right
    <  : Less than: True if the left operand is less than the right
    == : Equal to: True if both operands are equal
    != : Not equal to – True if operands are not equal
    >= : Greater than or equal to True if the left operand is greater than or equal to the right
    <= : Less than or equal to True if the left operand is less than or equal to the right

    and : Logical AND: True if both the operands are true
    or  : Logical OR: True if either of the operands is true 
    not : Logical NOT: True if the operand is false 

    is  : True if the operands are identical 
    is not : True if the operands are not identical 

    & : Bitwise AND
    | : Bitwise OR
    ~ : Bitwise NOT
    ^ : Bitwise XOR
    >> : Bitwise right shift
    << : Bitwise left shift
```
