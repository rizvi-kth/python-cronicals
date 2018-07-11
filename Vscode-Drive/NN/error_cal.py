from sklearn.metrics import mean_squared_error
print('Input: %s' % input_data)
print('Weight 0: %s' % weights_0)
print('Weight 1: %s' % weights_1)

 
# Create model_output_0 
model_output_0 = []
# Create model_output_0
model_output_1 = []

# Loop over input_data
for row in input_data:
    # Append prediction to model_output_0
    model_output_0.append(predict_with_network(row, weights_0 ))
    
    # Append prediction to model_output_1
    model_output_1.append(predict_with_network(row, weights_1 ))

# Calculate the mean squared error for model_output_0: mse_0
mse_0 = mean_squared_error(target_actuals, model_output_0)

# Calculate the mean squared error for model_output_1: mse_1
mse_1 = mean_squared_error(target_actuals, model_output_1)

# Print mse_0 and mse_1
print("Mean squared error with weights_0: %f" %mse_0)
print("Mean squared error with weights_1: %f" %mse_1)
