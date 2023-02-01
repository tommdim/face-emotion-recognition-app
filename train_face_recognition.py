#wrapping our function into tf function that compiles a functionn into a callable TensorFlow graph
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from tensorflow.keras.models import Model

class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
    
def make_embedding(): 
    inp = Input(shape=(100,100,3), name='input_image')
    
    # First block
    c1 = Conv2D(64, (10,10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
    
    # Second block
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
    
    # Third block 
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)
    
    # Final embedding block
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    
    
    return Model(inputs=[inp], outputs=[d1], name='embedding')


def make_siamese_model(): 
    embedding = make_embedding()
    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(100,100,3))
    
    # Validation image in the network 
    validation_image = Input(name='validation_img', shape=(100,100,3))
    
    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))
    
    # Classification layer 
    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


@tf.function
def train_step(batch, siamese_model):
    
    # Record all of our operations 
    with tf.GradientTape() as tape:     
        # Get anchor and positive/negative image
        X = batch[:2]  #
        # Get label
        y = batch[2]
        
        # Forward pass
        yhat = siamese_model(X, training=True)  #passing a feature, training is equal to true in order to activate layers
        # Calculate loss
        binary_cross_loss = tf.losses.BinaryCrossentropy()
        loss = binary_cross_loss(y, yhat) #pass the true value and the prediction
    print(loss)
        
    # Calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    
    # Calculate updated weights and apply to siamese model
    opt = tf.keras.optimizers.Adam(1e-4)
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
        
    # Return loss
    return loss

# Import metric calculations
from tensorflow.keras.metrics import Precision, Recall

def train(data, EPOCHS):
    siamese_model = make_siamese_model()
    # Loop through epochs
    for epoch in range(1, EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbr(len(data))
        
        # Creating a metric object 
        r = Recall()
        p = Precision()
        
        # Loop through each batch
        for idx, batch in enumerate(data):
            # Run train step here
            loss = train_step(batch, siamese_model)
            yhat = siamese_model.predict(batch[:2])
            r.update_state(batch[2], yhat)
            p.update_state(batch[2], yhat) 
            progbar.update(idx+1)
        print(loss.numpy(), r.result().numpy(), p.result().numpy())

def make_model(train_data,nome_modello, num_epoch,test_data,siamese_model):      
    train(train_data, num_epoch)
    test_input, test_val, y_true = test_data.as_numpy_iterator().next()

    y_hat = siamese_model.predict([test_input, test_val])
    # Post processing the results 
    [1 if prediction > 0.5 else 0 for prediction in y_hat ]
    # Creating a metric object 
    m = Recall()
    # Calculating the recall value 
    m.update_state(y_true, y_hat)
    # Return Recall Result
    m.result().numpy()
    # Creating a metric object 
    m = Precision()
    # Calculating the recall value 
    m.update_state(y_true, y_hat)
    # Return Recall Result
    m.result().numpy()
    r = Recall()
    p = Precision()
    for test_input, test_val, y_true in test_data.as_numpy_iterator():
        yhat = siamese_model.predict([test_input, test_val])
        r.update_state(y_true, yhat)
        p.update_state(y_true,yhat) 

    print(r.result().numpy(), p.result().numpy())
    #Save weights (model)
    siamese_model.save(nome_modello)

