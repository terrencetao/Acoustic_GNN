from build_graph import *
import pandas as pd
import tensorflow as tf
from sklearn import svm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', help ='source folder')
    #parser.add_argument('--output_folder', help ='output folder')
    args = parser.parse_args()
    
    input_folder = args.input_folder
    #dataset =  pd.read_csv(os.path.join(input_folder,'dataset.csv'), dtype='object')
    dataset =  pd.read_pickle(os.path.join(input_folder,'dataset.pl'))
   
# Create an edges array (sparse adjacency matrix) of shape [2, num_edges].
    if os.path.exists(os.path.join(input_folder,'edges.npy')) and  os.path.exists(os.path.join(input_folder,'weights.npy')):
        with open(os.path.join(input_folder,'edges.npy'), 'rb') as f:
            edges = np.load(f, allow_pickle=True)
        with open(os.path.join(input_folder,'weights.npy'), 'rb') as f:
            edge_weights = np.load(f, allow_pickle=True)
    else:
        edges, edge_weights = build_edges(dataset)
        with open(os.path.join(input_folder,'edges.npy'), 'wb') as f:
            np.save(f, edges)
        with open(os.path.join(input_folder,'weights.npy'), 'wb') as f:
            np.save(f,edge_weights)
    edges = edges.T
    #edges = build_edge(dataset)
    print('edges buided')
# Create an edge weights array of ones.
    #edge_weights = build_edge_weights(edges,dataset)
    #edge_weights = tf.ones(shape=edges.shape[1])
    print('edge_weights buided')
# Create a node features array of shape [num_nodes, num_features].
    node_features = build_node(dataset)
    print('node features buided')
# Create graph info tuple with node_features, edges, and edge_weights.
    graph_info = (node_features, edges, edge_weights)

    print("Edges shape:", edges.shape)
    print("Nodes shape:", node_features.shape)
    print("edge_weights", edge_weights.shape)

    
    class_values = sorted(dataset["label"].unique())
    class_idx = {name: id for id, name in enumerate(class_values)}
    
    num_classes = len(pd.unique(dataset['label']))
    dataset["label"] = dataset["label"].apply(lambda value: class_idx[value])



    gnn_model = GNNNodeClassifier(
    graph_info=graph_info,
    num_classes=num_classes,
    hidden_units=hidden_units,
    conv_param=conv_param,
    pooling_param = pooling_param,
    dropout_rate=dropout_rate,
    name="gnn_model",
    )
    gnn_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    gnn_model.load_weights(os.path.join(input_folder,"best_weights/Weights"))

    test_data = pd.read_pickle(os.path.join(input_folder,"test_data.pl"))
    train_data = pd.read_pickle(os.path.join(input_folder,'train_data.pl'))
    # with open('node_repesentations.pkl', 'wb') as f:  # open a text file
    #     pl.load(embeddings)
   
    x_train = train_data.ids.to_numpy()
    y_train = train_data.label.to_numpy()

    x_test = test_data.ids.to_numpy()
    y_test = test_data.label.to_numpy()
    _, test_accuracy = gnn_model.evaluate(x=x_test, y=y_test, verbose=0)
    print(f"Test accuracy: {round(test_accuracy * 100, 2)}%")
    _, train_accuracy = gnn_model.evaluate(x=x_train, y=y_train, verbose=0)
    print(f"Train accuracy: {round(train_accuracy * 100, 2)}%")
    # print(tf.gather(embeddings, x_train).shape)


    # clf =  svm.SVC()
    # clf.fit(tf.gather(embeddings, x_train), y_train)
    # y_pred = clf.predict(tf.gather(embeddings, x_test))
    # print(precision_score(y_true=y_test, y_pred=y_pred,average='macro'))