# NumPy Tutorial
## Summary Slides

---

## Slide 1: What is Anaconda?

### Anaconda Distribution

**Anaconda** is a popular open-source distribution for Python and R programming, designed for scientific computing and data science.

### Key Features

- **Pre-installed Packages:** 250+ pre-installed data science packages
- **Package Manager:** `conda` - powerful package and environment manager
- **Cross-Platform:** Works on Windows, macOS, and Linux
- **Environment Management:** Create isolated environments for different projects
- **No Administrator Rights:** Can be installed without admin privileges

### Why Use Anaconda?

- Simplifies package installation and dependency management
- Avoids conflicts between different project requirements
- Includes essential libraries: NumPy, Pandas, Matplotlib, Scikit-learn
- Perfect for data science and machine learning workflows

**Official Website:** https://www.anaconda.com/

---

## Slide 2: What is Jupyter Notebook?

### Interactive Computing Environment

**Jupyter Notebook** is an open-source web application that allows you to create and share documents containing:
- Live code
- Equations (LaTeX)
- Visualizations
- Narrative text (Markdown)

### Key Features

- **Cell-based Execution:** Run code in small chunks (cells)
- **Inline Visualizations:** Plots and charts display directly in notebook
- **Multiple Languages:** Supports Python, R, Julia, and more
- **Rich Output:** Images, HTML, videos, custom objects
- **Shareable:** Export to HTML, PDF, slides

### Workflow Benefits

- **Exploratory Analysis:** Test ideas quickly
- **Documentation:** Code + explanations in one place
- **Reproducibility:** Share complete analysis workflow
- **Teaching:** Excellent for tutorials and education

---

## Slide 3: Environment Setup

### Step 1: Install Anaconda

1. Download from: https://www.anaconda.com/download
2. Choose your operating system (Windows/macOS/Linux)
3. Run the installer (recommended: add to PATH)
4. Verify installation: `conda --version`

### Step 2: Create Environment (Optional)

```bash
# Create new environment
conda create -n numpy-tutorial python=3.9

# Activate environment
conda activate numpy-tutorial
```

### Step 3: Install NumPy

```bash
# NumPy usually comes with Anaconda
# If needed, install with:
conda install numpy

# Or using pip
pip install numpy
```

### Step 4: Launch Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook

# Browser opens automatically at http://localhost:8888
```

### Step 5: Import NumPy

```python
import numpy as np
```

---

## Slide 4: Introduction - Why NumPy?

### The Data Science Challenge

All data can be fundamentally represented as **arrays of numbers:**
- **Digital images:** 2D arrays of pixels
- **Sound clips:** 1D arrays of intensity vs. time
- **Text:** Numerical representations (word frequency, embeddings)
- **Measurements:** Collections of numerical values

### The Process: Feature Engineering

**First step in data analysis:** Transform data into arrays of numbers

### Why NumPy?

- **Fundamental:** Core package for scientific computing in Python
- **High-performance:** Multidimensional array object
- **Comprehensive:** Tools for working with numerical arrays
- **Essential:** Storage and manipulation of numerical arrays is fundamental to data science

---

## Slide 5: NumPy Overview

### What is NumPy?

**NumPy** = **Num**eric **Py**thon (or **Num**erical **Py**thon)

### Core Features

- **Powerful N-dimensional array object**
- **Broadcasting functions** (sophisticated element-wise operations)
- **Linear algebra tools**
- **Fourier transform capabilities**
- **Random number generation**
- **Integration with C/C++ and Fortran code**

### Implementation

- Written in C
- Based on ATLAS library (linear algebra operations)
- Optimized for performance
- Much faster than pure Python

### Official Documentation

http://www.numpy.org/

---

## Slide 6: Python vs C - Type Systems

### Dynamically Typed (Python)

**Python:**
```python
counter = 0
for i in range(100):
    counter += i
```

**Flexibility:**
```python
a = 1       # integer
a = "one"   # now a string - no problem!
```

### Statically Typed (C)

**C:**
```c
int counter = 0;
for(int i=0; i<100; i++){
    counter += i;
}
```

**Type Safety:**
```c
int a = 1;
a = "one";  // COMPILATION ERROR!
```

### Trade-off

**Python:** Convenient and flexible, but has overhead
**C:** Fast and efficient, but less flexible

---

## Slide 7: Python Data Type Overhead

### Python Integer Structure (C Implementation)

```c
struct _longobject {
    long ob_refcnt;        // Reference count
    PyTypeObject *ob_type;  // Type of variable
    size_t ob_size;         // Size of data members
    long ob_digit[1];       // Actual integer value
};
```

### Four Components for Single Integer

1. **ob_refcnt:** Reference count
2. **ob_type:** Type information
3. **ob_size:** Size information
4. **ob_digit:** Actual value

### Comparison

**C Integer:**
- Direct memory pointer to bytes encoding integer value
- Minimal overhead

**Python Integer:**
- Pointer to complex structure with metadata
- Flexibility comes at a cost

---

## Slide 8: Python Lists vs NumPy Arrays

### Python Lists

**Definition:**
```python
pythonList = list(range(5))
```

**Mixed Types Allowed:**
```python
pythonList = [True, "2", 3.0, 4]
```

**Problem:**
- Each item stores its own type info, reference count, etc.
- Lots of redundant information
- Inefficient for numerical operations

---

### NumPy Solution: Fixed-Type Arrays

**Key Idea:** Store data in fixed-type arrays

**Efficiency:**
- Much less overhead
- Faster computation
- Optimized for numerical operations

**Python's Built-in Array Module (Python 3.3+):**
```python
import array
L = list(range(10))
A = array.array('i', L)
```

**NumPy Goes Further:**
- Efficient storage (like array module)
- **Plus:** Efficient operations on data

---

## Slide 9: NumPy Array Structure

### NumPy Array Components

A NumPy array is basically **a collection of pointers:**

1. **Data Pointer:** Memory address of first byte
2. **Data Type (dtype):** Kind of elements (int, float, etc.)
3. **Shape:** Dimensions of the array
4. **Strides:** Bytes to skip to get to next element

### C Implementation

```c
typedef struct PyArrayObject {
    PyObject_HEAD

    char *data;              // Block of memory
    PyArray_Descr *descr;    // Data type descriptor
    int nd;                  // Number of dimensions
    npy_intp *dimensions;    // Shape
    npy_intp *strides;       // Indexing scheme

    PyObject *base;
    int flags;
    PyObject *weakreflist;
} PyArrayObject;
```

### Key Insight

NumPy stores **how to locate and interpret elements** efficiently

---

## Slide 10: Creating First NumPy Array

### Import NumPy

```python
import numpy as np
```

### Create Array of Zeros

```python
# Create a length-10 integer array filled with zeros
np.zeros(10, dtype=int)
```

**Output:**
```
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
```

### Specify Data Type

**String notation:**
```python
np.zeros(10, dtype='int16')
```

**NumPy object notation:**
```python
np.zeros(10, dtype=np.int16)
```

**Both produce:**
```
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int16)
```

---

## Slide 11: NumPy Standard Data Types

### Integer Types

| Type | Description | Range |
|------|-------------|-------|
| **int8** | Byte | -128 to 127 |
| **int16** | Integer | -32768 to 32767 |
| **int32** | Integer | -2147483648 to 2147483647 |
| **int64** | Integer | -9223372036854775808 to 9223372036854775807 |
| **uint8** | Unsigned | 0 to 255 |
| **uint16** | Unsigned | 0 to 65535 |
| **uint32** | Unsigned | 0 to 4294967295 |
| **uint64** | Unsigned | 0 to 18446744073709551615 |

### Floating Point Types

| Type | Description |
|------|-------------|
| **float16** | Half precision: sign bit, 5 bits exp, 10 bits mantissa |
| **float32** | Single precision: sign bit, 8 bits exp, 23 bits mantissa |
| **float64** | Double precision: sign bit, 11 bits exp, 52 bits mantissa |

---

### Boolean and Complex Types

| Type | Description |
|------|-------------|
| **bool_** | Boolean (True/False) stored as byte |
| **complex64** | Two 32-bit floats (real + imaginary) |
| **complex128** | Two 64-bit floats (real + imaginary) |

### Default Types

| Type | Description |
|------|-------------|
| **int_** | Default integer (same as C long) |
| **intp** | Integer for indexing (C ssize_t) |
| **float_** | Shorthand for float64 |
| **complex_** | Shorthand for complex128 |

---

## Slide 12: Creating NumPy Arrays

### Random Arrays

```python
np.random.seed(7)  # Set seed for reproducibility

a1 = np.random.randint(10, size=6)         # 1D array
a2 = np.random.randint(10, size=(3, 4))    # 2D array
a3 = np.random.randint(10, size=(3, 4, 5)) # 3D array
```

**Output:**
```python
[4 9 6 3 3 7]

[[7 9 7 8]
 [9 8 7 6]
 [4 0 7 0]]
```

---

### From Python Lists

```python
a4 = np.array([1, 2, 3])              # 1D array
a5 = np.array([[1,2,3],[4,5,6]])      # 2D array
```

**Output:**
```python
[1 2 3]

[[1 2 3]
 [4 5 6]]
```

---

## Slide 13: Array Creation Functions

### Common Creation Functions

```python
# All zeros
a6 = np.zeros((2,2))

# All ones
a7 = np.ones((1,2))

# Constant value
a8 = np.full((2,2), 7)

# Identity matrix
a9 = np.eye(2)

# Evenly spaced values
a10 = np.linspace(0, 100, 6)  # 6 values from 0 to 100

# Range with step
a11 = np.arange(0, 10, 3)      # [0, 3, 6, 9]

# Fill with specific value
a12 = np.full((2,3), 8)
```

---

### Output Examples

```python
[[0. 0.]    # zeros
 [0. 0.]]

[[1. 1.]]   # ones

[[7 7]      # constant
 [7 7]]

[[1. 0.]    # identity
 [0. 1.]]

[  0.  20.  40.  60.  80. 100.]  # linspace

[0 3 6 9]   # arange

[[8 8 8]    # full
 [8 8 8]]
```

---

## Slide 14: NumPy Array Attributes

### Essential Attributes

```python
a3 = np.random.randint(10, size=(3, 4, 5))

print("ndim:", a3.ndim)        # Number of dimensions
print("shape:", a3.shape)      # Size of each dimension
print("size:", a3.size)        # Total number of elements
print("dtype:", a3.dtype)      # Data type
print("itemsize:", a3.itemsize) # Size per element (bytes)
print("nbytes:", a3.nbytes)    # Total size (bytes)
```

**Output:**
```
ndim: 3
shape: (3, 4, 5)
size: 60
dtype: int32
itemsize: 4 bytes
nbytes: 240 bytes
```

---

### Attribute Summary

| Attribute | Description |
|-----------|-------------|
| **ndim** | Number of dimensions |
| **shape** | Size of each dimension |
| **size** | Total number of elements |
| **dtype** | Data type of elements |
| **itemsize** | Size in bytes per element |
| **nbytes** | Total size in bytes (size × itemsize) |

---

## Slide 15: Array Indexing

### One-Dimensional Arrays

Similar to Python list indexing:

```python
a1 = np.array([4, 9, 6, 3, 3, 7])

print(a1[0])   # First element: 4
print(a1[-1])  # Last element: 7
```

---

### Multi-Dimensional Arrays

Use **comma-separated tuple of indices:**

```python
a2 = np.array([[7, 9, 7, 8],
               [9, 8, 7, 6],
               [4, 0, 7, 0]])

print(a2[0, 0])    # Row 0, Column 0: 7
print(a2[2, 3])    # Row 2, Column 3: 0
```

---

### Modifying Values

```python
a2[0, 0] = 1111

print(a2)
```

**Output:**
```
[[1111    9    7    8]
 [   9    8    7    6]
 [   4    0    7    0]]
```

---

## Slide 16: Array Slicing

### Slicing Syntax

```
array[start:stop:step]
```

**Default values:**
- `start` = 0
- `stop` = size of dimension
- `step` = 1

---

### One-Dimensional Slicing

```python
x = np.arange(10)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

x[:2]    # First two elements: [0, 1]
x[2:]    # Elements after index 2: [2, 3, 4, 5, 6, 7, 8, 9]
x[4:7]   # Middle sub-array: [4, 5, 6]
x[::2]   # Every second element: [0, 2, 4, 6, 8]
```

---

### Multi-Dimensional Slicing

```python
x = np.array([[4, 5, 3, 0],
              [4, 8, 6, 7],
              [2, 7, 3, 8]])

x[:2, :3]   # Two rows, three columns
# [[4 5 3]
#  [4 8 6]]

x[:, 0]     # First column: [4, 4, 2]

x[0, :]     # First row: [4, 5, 3, 0]
```

---

## Slide 17: Array Views vs Copies

### Important Difference

**Python lists:** Slices return **copies**
**NumPy arrays:** Slices return **views** (not copies!)

### View Behavior

```python
x = np.array([[1, 2], [3, 4]])
sub = x[:1, :1]  # Get subarray (view)
sub[0, 0] = 999  # Modify view

print(x)  # Original array is MODIFIED!
# [[999   2]
#  [  3   4]]
```

---

### Creating Copies

Use **copy()** method for independent copy:

```python
x = np.array([[6, 6, 5, 6],
              [5, 7, 1, 5],
              [4, 4, 9, 9]])

x_copy = x[:2, :2].copy()  # Create copy
x_copy[0, 0] = 45          # Modify copy

print(x_copy)
# [[45  6]
#  [ 5  7]]

print(x)  # Original unchanged
# [[6 6 5 6]
#  [5 7 1 5]
#  [4 4 9 9]]
```

---

## Slide 18: Array Reshape

### Reshaping Arrays

Change array dimensions without changing data:

```python
x = np.array([1, 2, 3])
print(x)
# [1 2 3]

x = x.reshape(3, 1)  # 3 rows, 1 column
print(x)
# [[1]
#  [2]
#  [3]]
```

---

### Transposing Arrays

**Transpose:** Rows become columns and vice versa

```python
x = np.array([[1, 2],
              [3, 4]])

print(x)
# [[1 2]
#  [3 4]]

print(x.T)  # Transpose
# [[1 3]
#  [2 4]]
```

**Use case:** Essential for matrix operations and data manipulation

---

## Slide 19: Array Concatenation

### One-Dimensional Concatenation

```python
x = np.array([1, 2, 3])
y = np.array([3, 2, 1])

np.concatenate([x, y])
# array([1, 2, 3, 3, 2, 1])
```

---

### Two-Dimensional Concatenation

```python
grid = np.array([[1, 2, 3],
                 [4, 5, 6]])

np.concatenate([grid, grid])
# array([[1, 2, 3],
#        [4, 5, 6],
#        [1, 2, 3],
#        [4, 5, 6]])
```

---

## Slide 20: Vertical and Horizontal Stacking

### Vertical Stack (vstack)

```python
x = np.array([1, 2, 3])
grid = np.array([[9, 8, 7],
                 [6, 5, 4]])

np.vstack([x, grid])
# array([[1, 2, 3],
#        [9, 8, 7],
#        [6, 5, 4]])
```

---

### Horizontal Stack (hstack)

```python
grid = np.array([[9, 8, 7],
                 [6, 5, 4]])
y = np.array([[99],
              [99]])

np.hstack([grid, y])
# array([[ 9,  8,  7, 99],
#        [ 6,  5,  4, 99]])
```

---

### Depth Stack (dstack)

**dstack** stacks arrays along the **third axis** (depth)

Useful for combining 2D arrays into 3D structures (e.g., RGB channels)

---

## Slide 21: Array Splitting

### Split Function

```python
x = [1, 2, 3, 4, 5, 6, 7, 8]
x1, x2, x3 = np.split(x, [3, 5])

print(x1, x2, x3)
# [1 2 3] [4 5] [6 7 8]
```

**Syntax:** `np.split(array, [split_points])`

---

### Horizontal and Vertical Splits

**hsplit:** Split horizontally (column-wise)
```python
grid = np.array([[1, 2, 3, 4],
                 [5, 6, 7, 8]])
left, right = np.hsplit(grid, [2])
```

**vsplit:** Split vertically (row-wise)
```python
grid = np.array([[1, 2],
                 [3, 4],
                 [5, 6]])
top, bottom = np.vsplit(grid, [2])
```

---

## Slide 22: Array Math - Element-wise Operations

### Addition

```python
x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)

# Operator
print(x + y)

# Function
print(np.add(x, y))

# Both produce:
# [[ 6.  8.]
#  [10. 12.]]
```

---

### Subtraction

```python
print(x - y)
print(np.subtract(x, y))

# [[-4. -4.]
#  [-4. -4.]]
```

---

### Multiplication (Element-wise)

```python
print(x * y)
print(np.multiply(x, y))

# [[ 5. 12.]
#  [21. 32.]]
```

---

### Division

```python
print(x / y)
print(np.divide(x, y))

# [[0.2        0.33333333]
#  [0.42857143 0.5       ]]
```

---

### Square Root

```python
print(np.sqrt(x))

# [[1.         1.41421356]
#  [1.73205081 2.        ]]
```

---

## Slide 23: Matrix Operations

### Dot Product / Matrix Multiplication

**Important:** Use **dot** for matrix multiplication, NOT `*`

```python
x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)

# Method 1
print(x.dot(y))

# Method 2
print(np.dot(x, y))

# Both produce:
# [[19. 22.]
#  [43. 50.]]
```

**Note:** `*` does element-wise multiplication, `dot` does matrix multiplication

---

## Slide 24: Aggregation Functions

### Sum

```python
x = np.array([[1, 2], [3, 4]], dtype=np.float64)

np.sum(x)           # Sum of all elements: 10.0
np.sum(x, axis=0)   # Sum of each column: [4. 6.]
np.sum(x, axis=1)   # Sum of each row: [3. 7.]
```

---

### Min and Max

```python
np.min(x)   # Minimum: 1.0
np.max(x)   # Maximum: 4.0
```

---

### Other Useful Aggregations

```python
np.mean(x)      # Mean (average)
np.std(x)       # Standard deviation
np.var(x)       # Variance
np.median(x)    # Median
np.argmin(x)    # Index of minimum
np.argmax(x)    # Index of maximum
```

**Axis parameter:**
- `axis=0`: Column-wise operation
- `axis=1`: Row-wise operation
- No axis: Operation on entire array

---

## Slide 25: Broadcasting - Introduction

### The Problem

**Strict rule:** Array arithmetic requires **same shape**

```python
a = np.array([0, 1, 2])
b = np.array([5, 5, 5])
a + b  # Works: both shape (3,)
# array([5, 6, 7])
```

**This should fail:**
```python
a = np.array([0, 1, 2])
a + 5  # Different shapes: (3,) vs scalar
```

**But it doesn't!** Thanks to **broadcasting**

---

### What is Broadcasting?

**Broadcasting:** NumPy's method to perform arithmetic on arrays with **different shapes**

**How it works:** NumPy "stretches" the smaller array to match the larger one

```python
a = np.array([0, 1, 2])
a + 5
# array([5, 6, 7])
```

**Conceptually:**
```
    [0, 1, 2]
+   [5, 5, 5]  # 5 is broadcast to [5, 5, 5]
=   [5, 6, 7]
```

---

## Slide 26: Broadcasting Examples

### 1D + 2D Arrays

```python
a = np.array([0, 1, 2])      # Shape: (3,)
b = np.ones((3, 3))           # Shape: (3, 3)

result = b + a                # Shape: (3, 3)

# [[1. 2. 3.]
#  [1. 2. 3.]
#  [1. 2. 3.]]
```

**What happened:**
- `a` broadcasted to shape (3, 3)
- Each row of `b` gets `a` added to it

---

### Complex Broadcasting

```python
a = np.arange(3)              # [0, 1, 2], shape: (3,)
b = np.arange(3)[:, np.newaxis]  # [[0], [1], [2]], shape: (3, 1)

print(a)
# [0 1 2]

print(b)
# [[0]
#  [1]
#  [2]]

c = a + b
print(c)
# [[0 1 2]
#  [1 2 3]
#  [2 3 4]]
```

---

### Broadcasting Visualization

```
    a:  [0  1  2]     Shape: (3,)

    b:  [[0]          Shape: (3, 1)
         [1]
         [2]]

Broadcast to:

    a:  [[0  1  2]    Shape: (3, 3)
         [0  1  2]
         [0  1  2]]

    b:  [[0  0  0]    Shape: (3, 3)
         [1  1  1]
         [2  2  2]]

Result: [[0  1  2]
         [1  2  3]
         [2  3  4]]
```

---

## Slide 27: Broadcasting Rules

### Broadcasting Rules (Simplified)

Two dimensions are **compatible** when:
1. They are **equal**, OR
2. One of them is **1**

### Broadcasting Process

1. If arrays have different dimensions, **pad** the smaller shape with 1s on the left
2. If dimensions don't match, **stretch** dimension with size 1 to match the other

### Examples

✓ **Compatible:**
```
(3, 4) and (3, 4)  # Same shape
(3, 4) and (4,)    # Becomes (3, 4) and (1, 4) → (3, 4)
(3, 1) and (1, 4)  # → (3, 4)
```

✗ **Incompatible:**
```
(3, 4) and (3,)    # Can't align
(3, 4) and (2, 4)  # 3 ≠ 2 and neither is 1
```

---

## Slide 28: Summary - Key Concepts

### NumPy Core Concepts

| Concept | Description |
|---------|-------------|
| **ndarray** | N-dimensional array object (core data structure) |
| **dtype** | Data type of array elements |
| **shape** | Dimensions of the array |
| **Broadcasting** | Arithmetic on arrays of different shapes |
| **Vectorization** | Operations on entire arrays (no loops needed) |
| **View** | Array slice that references original data |
| **Copy** | Independent duplicate of array data |
| **Axis** | Dimension along which operation is performed |

---

### Why NumPy?

✓ **Performance:** 10-100× faster than pure Python
✓ **Convenience:** No explicit loops needed
✓ **Memory Efficient:** Fixed-type arrays
✓ **Ecosystem:** Foundation for pandas, scikit-learn, TensorFlow, etc.

---

## Slide 29: Common NumPy Operations Cheat Sheet

### Array Creation

```python
np.array([1, 2, 3])           # From list
np.zeros((2, 3))              # All zeros
np.ones((2, 3))               # All ones
np.full((2, 3), 7)            # Constant value
np.eye(3)                     # Identity matrix
np.arange(0, 10, 2)           # Range: [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)          # 5 values: [0, 0.25, 0.5, 0.75, 1]
np.random.random((2, 3))      # Random [0, 1)
np.random.randint(0, 10, (2, 3))  # Random integers
```

---

### Array Inspection

```python
arr.shape                     # Dimensions
arr.ndim                      # Number of dimensions
arr.size                      # Total number of elements
arr.dtype                     # Data type
```

---

### Array Manipulation

```python
arr.reshape(2, 3)             # Reshape
arr.T                         # Transpose
arr.flatten()                 # Flatten to 1D
np.concatenate([a, b])        # Join arrays
np.split(arr, 3)              # Split array
```

---

### Mathematical Operations

```python
np.add(a, b)      # a + b
np.subtract(a, b) # a - b
np.multiply(a, b) # Element-wise: a * b
np.divide(a, b)   # a / b
np.dot(a, b)      # Matrix multiplication
np.sqrt(a)        # Square root
np.exp(a)         # Exponential
np.log(a)         # Natural log
```

---

### Aggregations

```python
np.sum(arr)       # Sum
np.mean(arr)      # Mean
np.std(arr)       # Standard deviation
np.min(arr)       # Minimum
np.max(arr)       # Maximum
np.argmin(arr)    # Index of minimum
np.argmax(arr)    # Index of maximum
```

Add `axis=0` or `axis=1` for column/row operations

---

## Slide 30: NumPy Documentation & Resources

### Official Documentation

**NumPy Reference:** https://docs.scipy.org/doc/numpy/reference/

**User Guide:** https://numpy.org/doc/stable/user/

### Key Topics to Explore

- **Advanced indexing** (boolean indexing, fancy indexing)
- **Structured arrays** (record arrays)
- **Memory layout** (C vs Fortran order)
- **ufuncs** (universal functions)
- **Linear algebra** (np.linalg)
- **Random sampling** (np.random)
- **FFT** (Fast Fourier Transform)
- **Polynomial fitting**

---

### Learning Resources

**Books:**
- "Python Data Science Handbook" by Jake VanderPlas
- "NumPy Beginner's Guide" by Ivan Idris

**Online Tutorials:**
- NumPy Quickstart Tutorial (official)
- Real Python NumPy tutorials
- DataCamp NumPy courses

**Practice:**
- NumPy exercises (GitHub repositories)
- Kaggle notebooks
- Project Euler problems

---

## Slide 31: NumPy Best Practices

### Performance Tips

✓ **Vectorize operations** (avoid Python loops)
```python
# Slow
result = []
for x in arr:
    result.append(x ** 2)

# Fast
result = arr ** 2
```

✓ **Preallocate arrays** when possible
```python
# Good
result = np.zeros((1000, 1000))
# Fill result...
```

✓ **Use appropriate data types** (float32 vs float64)

---

### Code Clarity

✓ **Use meaningful array shapes**
```python
# Clear
image = np.zeros((height, width, 3))  # RGB image

# Unclear
data = np.zeros((100, 200, 3))
```

✓ **Document axis meanings**
```python
# axis=0: samples
# axis=1: features
np.mean(data, axis=1)  # Mean per sample
```

---

### Common Pitfalls

✗ **Views vs Copies confusion**
```python
sub = arr[:5]     # View - modifies original!
sub = arr[:5].copy()  # Copy - safe
```

✗ **Broadcasting errors**
```python
# Check shapes before operations
print(a.shape, b.shape)
```

✗ **Integer division**
```python
a = np.array([1, 2, 3])
a / 2          # float: [0.5, 1., 1.5]
a // 2         # int: [0, 1, 1]
```

---

## Slide 32: Next Steps

### What We Covered

✓ NumPy basics and motivation
✓ Array creation and attributes
✓ Indexing and slicing
✓ Reshaping and transposing
✓ Concatenation and splitting
✓ Mathematical operations
✓ Broadcasting

### What to Learn Next

1. **Pandas** - Data manipulation with DataFrames
2. **Matplotlib** - Data visualization
3. **SciPy** - Scientific computing (optimization, integration, etc.)
4. **Scikit-learn** - Machine learning
5. **Advanced NumPy** - Linear algebra, FFT, polynomial fitting

---

### Practice Exercises

1. Create a 10×10 array with values 1-100
2. Extract all even numbers from an array
3. Normalize an array (mean=0, std=1)
4. Create a checkerboard pattern (8×8)
5. Find indices where array values > threshold
6. Implement moving average using NumPy
7. Create a correlation matrix
8. Solve a system of linear equations

---

## Thank You!

### Key Takeaways

✓ **NumPy is fundamental** to scientific Python
✓ **Arrays are efficient** - use them instead of lists for numerical data
✓ **Vectorization is powerful** - avoid explicit loops
✓ **Broadcasting enables flexibility** - operations on different shapes
✓ **Understanding views vs copies** prevents bugs

### Keep Exploring!

**NumPy is the foundation for:**
- Data Science (Pandas)
- Machine Learning (Scikit-learn, TensorFlow, PyTorch)
- Scientific Computing (SciPy)
- Image Processing (OpenCV)
- And much more!

---

### Questions?

**Happy Computing with NumPy!**

*Remember: The best way to learn NumPy is to use it in real projects!*
