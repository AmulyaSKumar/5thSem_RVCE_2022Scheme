# Hill Climbing Algorithm

## Introduction
Hill Climbing is an optimization algorithm used to find the maximum value of a function. It starts with an initial solution and iteratively makes small changes to improve the result until no further improvements can be made. This algorithm is commonly used in artificial intelligence, robotics, and function optimization.

## Code Implementation
```python
def hill_climbing(func, start, step_size=0.01, max_iterations=1000):
    current_position = start
    current_value = func(current_position)

    for i in range(max_iterations):
        next_position_positive = current_position + step_size
        next_value_positive = func(next_position_positive)

        next_position_negative = current_position - step_size
        next_value_negative = func(next_position_negative)

        if next_value_positive > current_value and next_value_positive >= next_value_negative:
            current_position = next_position_positive
            current_value = next_value_positive
        elif next_value_negative > current_value and next_value_negative > next_value_positive:
            current_position = next_position_negative
            current_value = next_value_negative
        else:
            break

    return current_position, current_value

# Get the function from the user
while True:
    func_str = input("\nEnter a function of x: ")
    try:
        x = 0  # Test the function with a dummy value
        eval(func_str)
        break
    except Exception as e:
        print(f"Invalid function. Please try again. Error: {e}")

# Convert the string into a function
func = lambda x: eval(func_str)

# Get the starting point from the user
while True:
    start_str = input("\nEnter the starting value to begin the search: ")
    try:
        start = float(start_str)
        break
    except ValueError:
        print("Invalid input. Please enter a number.")

maxima, max_value = hill_climbing(func, start)
print(f"The maxima is at x = {maxima}")
print(f"The maximum value obtained is {max_value}")
```

## Sample Output
```
Enter a function of x: -x**2 + 4*x + 10
Enter the starting value to begin the search: 0
The maxima is at x = 2.0
The maximum value obtained is 14.0
```
```
Enter a function of x: -1*(x-3)**2 + 9
Enter the starting value to begin the search: 0
The maxima is at x = 3.000000000000001
The maximum value obtained is 9.0
```

## Advantages of Hill Climbing
1. **Efficiency**: Simple and fast for small problems.
2. **No Requirement for Large Memory**: Uses only local information.
3. **Versatility**: Can be used for various optimization problems.

## Disadvantages of Hill Climbing
1. **Local Maxima**: Can get stuck at a local maximum instead of the global maximum.
2. **Plateau Problem**: If all neighboring solutions have the same value, it can stop prematurely.
3. **Ridge Problem**: Can struggle with narrow paths in the solution space.

## Applications of Hill Climbing
- **Artificial Intelligence**: Used in decision-making and game playing.
- **Robotics**: Helps in motion planning and pathfinding.
- **Function Optimization**: Used to find the best parameters in machine learning.
- **Job Scheduling**: Optimizes resource allocation.

## Viva Questions and Answers
1. **What is Hill Climbing in AI?**
   - Hill Climbing is an optimization algorithm that iteratively improves a solution by making local changes to maximize (or minimize) an objective function.

2. **What is the main disadvantage of Hill Climbing?**
   - It can get stuck in local maxima and does not guarantee finding the global optimum.

3. **How can we overcome the local maxima problem in Hill Climbing?**
   - Using techniques like random restarts, simulated annealing, or adding some randomness to the step selection.

4. **What are the different types of Hill Climbing?**
   - Simple Hill Climbing, Steepest-Ascent Hill Climbing, and Stochastic Hill Climbing.

5. **How does Hill Climbing differ from Simulated Annealing?**
   - Simulated Annealing allows downward moves to escape local maxima, while Hill Climbing only moves in the direction of improvement.
