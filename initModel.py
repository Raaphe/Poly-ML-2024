import torch.nn.functional as F
from torch import nn
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm
from sklearn.metrics import r2_score

# Load data
df = pd.read_csv("dataset/participant_data.csv")

# Drop unnecessary columns
df = df.drop(["agentID", "Altitude", "Longitude", "Latitude"], axis=1)

# Convert '_time' to datetime and extract features
df['datetime'] = pd.to_datetime(df['_time'], errors='coerce')

df['hour_sin'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['datetime'].dt.hour / 24)

df['dayOfWeek_sin'] = np.sin(2 * np.pi * df['datetime'].dt.day_of_week / 7)
df['dayOfWeek_cos'] = np.cos(2 * np.pi * df['datetime'].dt.day_of_week / 7)

df['month_sin'] = np.sin(2 * np.pi * df['datetime'].dt.month / 12)
df['month_cos'] = np.cos(2 * np.pi * df['datetime'].dt.month / 12)

# Drop the original time column and rows with NaN datetime
df = df.drop(["_time"], axis=1)
df = df.drop(["datetime"], axis=1)

# Fill NaN values with the median
columns_with_nan = df.columns[df.isnull().any()]
for column in columns_with_nan:
    median_value = df[column].median()  
    df[column].fillna(median_value, inplace=True)  

# Convert non-numeric columns to numeric types
df = df.apply(pd.to_numeric, errors='coerce')

# Optionally save the cleaned DataFrame
df.to_csv("./new.csv", index=False)

# Convert to NumPy array and then to a PyTorch tensor
data = torch.from_numpy(df.to_numpy()).float()

# ============================== END DATA PREPROCESSING ============================== 

def create_inout_sequences(data, temporal_features, outpout_count, batch_size):
    sequences = []
    targets = []
    L = len(data)
    for i in range(L/batch_size):
        train_seq = data[i:i + temporal_features, :]  # Include all features for input
        train_label = data[i + temporal_features:i + temporal_features + outpout_count, :4]  # Target columns for latencies
        sequences.append(train_seq)
        targets.append(train_label)
    return torch.stack(sequences), torch.stack(targets)

# Adjust input and output sizes
input_size = 2  # e.g., looking at 2 past time steps
output_size = 4  # Predict 4 latency values

# Create sequences
X, y = create_inout_sequences(data, input_size, output_size)


# Reshape X to (num_samples, input_size, num_features)
X = X.permute(0, 2, 1)  # Change shape to (batch_size, features, time_steps)

# Now split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X Shape ");
print(X.shape);
print("Y Shape");
print(y.shape);

print("X_train shape");
print(X_train.shape);
print("y_train shape");
print(y_train.shape);



class MultiStepTCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        super(MultiStepTCN, self).__init__();

        self.tcn = nn.Sequential(
            nn.Conv1d(input_size, num_channels[0], kernel_size, padding=(kernel_size-1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels[0], num_channels[1], kernel_size, padding=(kernel_size-1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels[1], output_size, kernel_size, padding=(kernel_size-1))
        )
    
    def forward(self, x):
        # Input shape: (batch_size, time_steps, features)
        x = x.transpose(1, 2)  # Switch to (batch_size, features, time_steps) for Conv1d
        out = self.tcn(x)
        out = out.transpose(1, 2)  # Switch back to (batch_size, time_steps, output_size)
        return out

# Instantiate the model
input_size = 6  # 4 latency values + 2 cyclic temporal features
output_size = 4  # Predict 4 latency values
num_channels = [512, 256]  # Example for two TCN layers

model = MultiStepTCN(input_size, output_size, num_channels)

# Define the loss function and optimizer
loss_fn = nn.MSELoss()  # Mean Squared Error loss for regression tasks
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epochs = 100
batch_size = 32

# Update training loop if necessary
for epoch in tqdm(range(epochs)):
    model.train()
    running_loss = 0.0
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]
        
        optimizer.zero_grad()  # Reset gradients
        
        # Forward pass
        outputs = model(X_batch)
        
        # Compute the loss
        loss = loss_fn(outputs.view(-1, output_size), y_batch.view(-1, output_size))  # Ensure same shape
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss:.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred = model(X_test).view(-1, output_size)  # Reshape for consistency
    y_test_reshaped = y_test.view(-1, output_size)

# Calculate R² score
r2 = r2_score(y_test_reshaped.numpy(), y_pred.numpy())
print(f'R² Score: {r2:.4f}')
