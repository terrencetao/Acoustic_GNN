from build_graph import *
import pandas as pd
import tensorflow as tf

def create_model(hidden_units, conv_param,pooling_param, dropout_rate, num_classes  ,name=None):
    fnn_layers = []
    fnn_layers.append(layers.Conv2D(kernel_size=conv_param[1], filters =conv_param[2], input_shape = conv_param[0], activation='relu', padding='same'))
    fnn_layers.append(layers.MaxPooling2D(pool_size=pooling_param[0]))
    fnn_layers.append(layers.Flatten())
    for units in hidden_units:
        fnn_layers.append(layers.BatchNormalization())
        fnn_layers.append(layers.Dropout(dropout_rate))
        fnn_layers.append(layers.Dense(units, activation=tf.nn.gelu))
    fnn_layers.append(layers.Dense(units=num_classes, name="logits"))
    return keras.Sequential(fnn_layers, name=name)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', help ='source folder')
    #parser.add_argument('--output_folder', help ='output folder')
    args = parser.parse_args()
    
    input_folder = args.input_folder
    #dataset =  pd.read_csv(os.path.join(input_folder,'dataset.csv'), dtype='object')
    dataset =  pd.read_pickle(os.path.join(input_folder,'dataset.pl'))
    node_features = build_node(dataset)

    test_data = pd.read_pickle(os.path.join(input_folder,"test_data.pl"))
    train_data = pd.read_pickle(os.path.join(input_folder,'train_data.pl'))
    

    x_train_idx = train_data.ids.to_numpy()
    x_train = tf.gather(node_features, x_train_idx)
    y_train = train_data.label.to_numpy()
    x_test_idx = test_data.ids.to_numpy()
    x_test = tf.gather(node_features, x_test_idx)
    y_test = test_data.label.to_numpy()
    
    num_classes = len(pd.unique(dataset['label']))
    cnn_model = create_model(hidden_units, conv_param,pooling_param, dropout_rate, num_classes  ,name=None)
    history = run_experiment(cnn_model, x_train, y_train, 'checkpoint_cnn')
    _, test_accuracy = cnn_model.evaluate(x=x_test, y=y_test, verbose=0)
    print(f"Test accuracy: {round(test_accuracy * 100, 2)}%")
    _, test_accuracy = cnn_model.evaluate(x=x_train, y=y_train, verbose=0)
    print(f"Train accuracy: {round(test_accuracy * 100, 2)}%")

    cnn_model.save_model(os.path.join(input_folder,'cnn_model'))
