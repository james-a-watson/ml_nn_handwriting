Hello, 

Thanks for checking out my MNIST GitHub project!

After studying ML for a little over a year and completing a 12 week course from Stanford University
I wanted to put the things I'd learnt into a neural network built in python. The development process 
was really fun and I enjoyed working through the issues that this sort of complex algorithm raises. 


# How the Neural Network Predicts
The neural network learns from 5000 labelled examples of handwritten numbers, each in a 20x20 pixel grid. 
This results in a 400 dimension input vector. The Network then takes the input vector and initialises 2 matricies
of weights that value certain pixels in the 20x20 grid more or less and based on which pixels are highlighted. 

Using a hidden layer of 25 nodes, and the two matricies, the network asigns a likelihood to the input grid 
containing each of the 10 digits 0-9. The number with the highest likelihood is chosen as the output for that 
learning example. After the network scores all 5000 learning examples it will calculate a cost that is essentially 
the score for that iteration. The higher the cost, the worse the network performed. 


# How the Neural Network Learns (Gradient Descent)
Once the cost has been calculated for an iteration the algorithm will take a small step down the "cost slope".
The gradient of the slope is calculated using linear algebra on the matrix of costs for the current iteration. 
A downward step is taken to reduce the cost in the next iteration. There is some level of randomness to how much
the cost is reduces but it will always go down. 

The system then uses backpropagation which takes the step down the cost slope and uses it to affect the 2 matricies 
of weights between the three layers of the network. for the next iteration the cost will be lower and therefore the 
predictions more accurate.

# Difficulties
The construction of the network was not the hardest part of this project. Beyond some fiddly numpy functions and 
the odd hour spent solving obscure array bugs, the concept and coding was within my competancies. The thing I 
learned the most in this project was how to optimize code. In no project I have built before were so many loops
required. I was calling the same function millions of times as the model iterated over each individual value.

Vectorization was the key to solving this. Rather than passing in all the output seperatly in a loop I made the adjustment
to pass in  vector into one function. That resulted in all the small calculations of the function happening at once to the
same vector and reducing the time spent computing the calculations significantly.s

When I first wrote the code it took just over 20 mins. I used vectorization to reduce the number of loops and 
cProfiling to identify which functions and methods were taking the most time. This allowed me to get the algorthm 
down to less than 1 minute for 100 iterations with 0 accuracy loss. An outcome of which I am very proud.
