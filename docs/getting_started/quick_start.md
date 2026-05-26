# **🚀 Quick Start**

Welcome to FluxRender! The core philosophy of this engine is simple: **you provide the math, and we handle the rest**.

To launch a fully interactive, GPU-accelerated visualization, you do not need to worry about rendering loops, UI layouts, or memory buffers. You only need to do two things:

1. Define a mathematical vector function.
2. Pass it to our high-level simulation wrapper.



## **1. Defining the Flow**
Your vector function must accept two numeric parameters (x and y) representing the spatial coordinates. It must return a tuple representing the vector's components at that point.

**Time Injection:** If your simulation is dynamic, simply add t as a third parameter (e.g., `def flow_vector(x, y, t):`). FluxRender's inspection engine will automatically detect it and inject the real-time simulation tick directly into your mathematical model!

## **2. The Minimal Setup**
Here is the absolute fastest way to see FluxRender in action. Create a new Python file and paste the following code:


<video autoplay loop muted playsinline width="100%">
  <source src="../../assets/FluxRender_quick_simulate.mp4" type="video/mp4">
</video>


```python
import FluxRender as fr
import numpy as np

# Define flow mathematics (vector function)
def flow_vector(x, y):
    X = np.sin(x) * y
    Y = np.cos(y) * x
    return X, Y

fr.quick_simulate(flow_vector) # This single line sets up a full interactive simulation with default settings.
```


## **3. Controls & Interaction**
Once the window opens, you are not just watching a static plot—you are inside a live mathematical laboratory. You can navigate the space and modify parameters on the fly:

 - `Right Mouse Button (Hold & Drag)`: Pan and move around the coordinate system.
 - `Q`: Zoom out.
 - `E`: Zoom in.
 - `Left Mouse Button`: Interact with the on-screen UI buttons to instantly switch rendering modes, change topological properties, or tweak scaling rules.


That's it! You have successfully built and run your first visualization. To learn how to customize colors, build your own interface, or combine multiple systems, move on to the Intermediate (Standard Scene) guide.



