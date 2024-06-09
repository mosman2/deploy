import tensorflow as tf
import mlflow
import mlflow.tensorflow
from mlflow.models.signature import infer_signature

def load_data():
    """Loads the MNIST dataset and normalizes it."""
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)

def build_model():
    """Builds a Sequential Keras model for digit classification."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
    return model

def train_model(model, x_train, y_train, epochs=5):
    """Trains the model on the training data."""
    model.fit(x_train, y_train, epochs=epochs)

def evaluate_model(model, x_test, y_test):
    """Evaluates the model on the test data and returns the test loss and accuracy."""
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_acc})
    return test_loss, test_acc

def make_predictions(model, x_test):
    """Makes predictions with the model on test data and returns softmax probabilities."""
    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])
    return probability_model(x_test[:5])

def main():
    # MLflow tracking
    mlflow.set_experiment("MNIST TensorFlow Experiment")
    mlflow.start_run()
    
    # Log parameters
    mlflow.log_param("model_type", "Sequential")
    mlflow.log_param("epochs", 5)
    
    # Load data
    (x_train, y_train), (x_test, y_test) = load_data()
    
    # Build the model
    model = build_model()
    mlflow.tensorflow.autolog()
    
    # Train the model
    train_model(model, x_train, y_train)
    
    # Evaluate the model
    test_loss, test_acc = evaluate_model(model, x_test, y_test)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    
    # Make predictions
    predictions = make_predictions(model, x_test)
    print("Predictions on the first 5 test samples:", predictions.numpy())
    
    # Infer the model signature
    signature = infer_signature(x_train, predictions)
    
    # Log the model with the inferred signature
    mlflow.keras.log_model(model, "model", signature=signature)
    
    mlflow.end_run()

if __name__ == "__main__":
    main()
