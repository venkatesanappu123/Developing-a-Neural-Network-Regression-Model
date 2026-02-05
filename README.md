# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
<img width="825" height="552" alt="image" src="https://github.com/user-attachments/assets/c03fd2c0-0ba3-4a4e-8f8f-8ac3f9ca692f" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:

### Register Number:

```
# Name : VENKATESAN R
# Register Number : 212224230299
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1, 8)
        self.fc2=nn.Linear(8, 10)
        self.fc3=nn.Linear(10, 1)
        self.relu=nn.ReLU()
        self.history={'loss':[]}

  def forward(self,x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x


# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(ai_brain.parameters(), lr=0.001)

# Name : VENKATESAN R
# Register Number : 212224230299
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    # Write your code here
    for epoch in range(epochs):
      optimizer.zero_grad()
      loss=criterion(ai_brain(X_train),y_train)
      loss.backward()
      optimizer.step()

      ai_brain.history['loss'].append(loss.item())
      if epoch % 200 == 0:
          print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')



```

### Dataset Information
<img width="194" height="353" alt="image" src="https://github.com/user-attachments/assets/3b058a1f-662e-4367-a1c5-1ad8a7f29151" />


### OUTPUT

### Training Loss Vs Iteration Plot
<img width="562" height="455" alt="image" src="https://github.com/user-attachments/assets/28f1ab9f-290f-4be7-bd41-e565e0bca4b6" />


### New Sample Data Prediction
<img width="910" height="122" alt="image" src="https://github.com/user-attachments/assets/2744fb58-1733-4888-8a93-3d606063a3f0" />/>


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
