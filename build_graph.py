import pandas as pd
import numpy as np 
import argparse 
import os
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle as pl




hidden_units = [32, 32]
conv_param =[(20,64,1), 32 ,3] #[input_shape, num_filter, kernel_size]
pooling_param = [2]# [pooling_size, stride, padding]
learning_rate = 0.01
dropout_rate = 0.5
num_epochs = 300
batch_size = 256





def distance(node_1, node_2):
    """
       input :
         node_1: matrix numpy (mfcc)
         node_2: matrix numpy (mfcc)
       purpose : compute distance between 2 nodes
       output : scalar
                distance
    """
    return np.linalg.norm(node_1-node_2)

def build_node (dataset):
    """
       input: DataFrame of data
       purpose: build node graph
       output: matrix [num_record, mfcc]
    """
    
    return np.array(list(dataset.sort_values("ids")['mfcc_feautres']))


# def build_edges(dataset, threshold =0, beta=0):
#     """
#      Principe of this fonction  followin logic:
#             1-compute a distance d between 2 nodes 
#                 if node1.label == node2:
#                     create edge 
#                     weights = d + beta 
#                 elif d>threshold :
#                      create edge
#                      weights = d
#         input: 
#           dataset :DataFrame of data
#           threshold: threshold at which distance is considered important enough to create the link
#           beta : the advantage given to the same-label node
#        purpose: build edge between 2 nodes and compute the weigths
#        output : edges =matrix [2, num_edges], weigths =[num_edges]
#     """
#     edges = []
#     weights = []
#     df_edges_w = []
#     ids = pd.unique(dataset['ids'])

#     for i,id1 in tqdm(enumerate(ids)):
#         j=i+1
#         while j<len(ids):
#             d = distance(dataset[dataset['ids']==id1]['mfcc_feautres'].values[0], dataset[dataset['ids']==ids[j]]['mfcc_feautres'].values[0])

#             if d>threshold:
#                 edges.append([id1, ids[j]])

#                 if dataset[dataset['ids']==id1]['label'].values[0]==dataset[dataset['ids']==ids[j]]['mfcc_feautres'].values[0]:
#                     weights.append(d+beta)
#                     weight = d+beta
#                 else:
#                     weights.append(d)
#                     weight = d
#                 df_edges_w.append([id,ids, weight])
#             j=j+1
#     pd.DataFrame(df_edges_w, columns = ['record_1', 'record_2', 'distance']).to_csv('edges_w.csv')


#     return np.array(edges), np.array(weights)






def build_edge(dataset):
    """
       input: 
          dataset :DataFrame of data
       purpose: build edge between 2 nodes which have a same label
       output : matrix [2, num_edges]
    """
    labels = pd.unique(dataset['label'])
    edges=[]
    for label in tqdm(labels):
        ids = list(dataset[dataset['label']==label]['ids'])
        for i, id in enumerate(ids):
            j=i+1
            while j<len(ids):
                edges.append([id, ids[j]])
                edges.append([ids[j],id])
                j=j+1
    pd.DataFrame(edges, columns = ['record_1', 'record_2']).to_csv('edges.csv')
    return np.array(edges)


def build_edge_weights (edges,dataset):
    """
       input : 
           edges: matrix of pair of ids (ids as index of record)
           dataset: dataframe content mfcc vector for record
       purpose : Assign a weigth to the edge between 2 nodes as a distance compute on mfcc values
       output : matrix [num_edges]
    """

    weights = []
    for edge in tqdm(edges):
        nd1, nd2 = edge
        weights.append(distance(dataset[dataset['ids']==nd1]['mfcc_feautres'].values[0], dataset[dataset['ids']==nd2]['mfcc_feautres'].values[0]))

    return np.array(weights)

def train_test_split(dataset, perc =0.5):
    train_data, test_data = [], []
    for _, group_data in dataset.groupby("label"):
        # Select around perc% of the dataset for training.
        random_selection = np.random.rand(len(group_data.index)) <= perc
        train_data.append(group_data[random_selection])
        test_data.append(group_data[~random_selection])

    train_data = pd.concat(train_data).sample(frac=1)
    test_data = pd.concat(test_data).sample(frac=1)

    return train_data, test_data




def run_experiment(model, x_train, y_train, checkpoint_folder='checkpoint_gnn'):
    # Compile the model.
    checkpoint_path = checkpoint_folder+"/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_freq=50*batch_size)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    # Create an early stopping callback.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_acc", patience=50, restore_best_weights=True
    )
    # Fit the model.
    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=[early_stopping,cp_callback],
    )

    return history

def display_learning_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history.history["loss"])
    ax1.plot(history.history["val_loss"])
    ax1.legend(["train", "test"], loc="upper right")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    ax2.plot(history.history["acc"])
    ax2.plot(history.history["val_acc"])
    ax2.legend(["train", "test"], loc="upper right")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    plt.show()

def create_ffn_first(hidden_units, conv_param,pooling_param, dropout_rate, name=None):
    fnn_layers = []
    fnn_layers.append(layers.Conv2D(kernel_size=conv_param[1], filters =conv_param[2], input_shape = conv_param[0], activation='relu', padding='same'))
    fnn_layers.append(layers.MaxPooling2D(pool_size=pooling_param[0]))
    fnn_layers.append(layers.Flatten())
    for units in hidden_units:
        fnn_layers.append(layers.BatchNormalization())
        fnn_layers.append(layers.Dropout(dropout_rate))
        fnn_layers.append(layers.Dense(units, activation=tf.nn.gelu))

    return keras.Sequential(fnn_layers, name=name)

def create_ffn_last(hidden_units, dropout_rate, name=None):
    fnn_layers = []

    for units in hidden_units:
        fnn_layers.append(layers.BatchNormalization())
        fnn_layers.append(layers.Dropout(dropout_rate))
        fnn_layers.append(layers.Dense(units, activation=tf.nn.gelu))

    return keras.Sequential(fnn_layers, name=name)



class GraphConvLayer(layers.Layer):
    def __init__(
        self,
        hidden_units,
        dropout_rate=0.2,
        aggregation_type="mean",
        combination_type="concat",
        normalize=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.normalize = normalize

        self.ffn_prepare = create_ffn_last(hidden_units,  dropout_rate)
        if self.combination_type == "gated":
            self.update_fn = layers.GRU(
                units=hidden_units,
                activation="tanh",
                recurrent_activation="sigmoid",
                dropout=dropout_rate,
                return_state=True,
                recurrent_dropout=dropout_rate,
            )
        else:
            self.update_fn = create_ffn_last(hidden_units, dropout_rate)

    def prepare(self, node_repesentations, weights=None):
        # node_repesentations shape is [num_edges, embedding_dim].
        messages = self.ffn_prepare(node_repesentations)
        if weights is not None:
            messages = messages * tf.expand_dims(weights, -1)
        return messages

    def aggregate(self, node_indices, neighbour_messages, node_repesentations):
        # node_indices shape is [num_edges].
        # neighbour_messages shape: [num_edges, representation_dim].
        # node_repesentations shape is [num_nodes, representation_dim]
        num_nodes = node_repesentations.shape[0]
        if self.aggregation_type == "sum":
            aggregated_message = tf.math.unsorted_segment_sum(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        elif self.aggregation_type == "mean":
            aggregated_message = tf.math.unsorted_segment_mean(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        elif self.aggregation_type == "max":
            aggregated_message = tf.math.unsorted_segment_max(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        else:
            raise ValueError(f"Invalid aggregation type: {self.aggregation_type}.")

        return aggregated_message

    def update(self, node_repesentations, aggregated_messages):
        # node_repesentations shape is [num_nodes, representation_dim].
        # aggregated_messages shape is [num_nodes, representation_dim].
        if self.combination_type == "gru":
            # Create a sequence of two elements for the GRU layer.
            h = tf.stack([node_repesentations, aggregated_messages], axis=1)
        elif self.combination_type == "concat":
            # Concatenate the node_repesentations and aggregated_messages.
            h = tf.concat([node_repesentations, aggregated_messages], axis=1)
        elif self.combination_type == "add":
            # Add node_repesentations and aggregated_messages.
            h = node_repesentations + aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}.")

        # Apply the processing function.
        node_embeddings = self.update_fn(h)
        if self.combination_type == "gru":
            node_embeddings = tf.unstack(node_embeddings, axis=1)[-1]

        if self.normalize:
            node_embeddings = tf.nn.l2_normalize(node_embeddings, axis=-1)
        return node_embeddings

    def call(self, inputs):
        """Process the inputs to produce the node_embeddings.

        inputs: a tuple of three elements: node_repesentations, edges, edge_weights.
        Returns: node_embeddings of shape [num_nodes, representation_dim].
        """

        node_repesentations, edges, edge_weights = inputs
        # Get node_indices (source) and neighbour_indices (target) from edges.
        node_indices, neighbour_indices = edges[0], edges[1]
        # neighbour_repesentations shape is [num_edges, representation_dim].
        neighbour_repesentations = tf.gather(node_repesentations, neighbour_indices)

        # Prepare the messages of the neighbours.
        neighbour_messages = self.prepare(neighbour_repesentations, edge_weights)
        # Aggregate the neighbour messages.
        aggregated_messages = self.aggregate(
            node_indices, neighbour_messages, node_repesentations
        )
        # Update the node embedding with the neighbour messages.
        return self.update(node_repesentations, aggregated_messages)

class GNNNodeClassifier(tf.keras.Model):
    def __init__(
        self,
        graph_info,
        num_classes,
        conv_param,
        pooling_param,
        hidden_units,
        aggregation_type="sum",
        combination_type="concat",
        dropout_rate=0.2,
        normalize=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Unpack graph_info to three elements: node_features, edges, and edge_weight.
        node_features, edges, edge_weights = graph_info
        self.node_features = tf.expand_dims(node_features, -1)
        self.edges = edges
        self.edge_weights = edge_weights
        # Set edge_weights to ones if not provided.
        if self.edge_weights is None:
            self.edge_weights = tf.ones(shape=edges.shape[1])
        # Scale edge_weights to sum to 1.
        self.edge_weights = self.edge_weights / tf.math.reduce_sum(self.edge_weights)


        # Create a process layer.
        self.preprocess = create_ffn_first(hidden_units, conv_param, pooling_param, dropout_rate, name="preprocess")
        # Create the first GraphConv layer.
        self.conv1 = GraphConvLayer(
            hidden_units,
            dropout_rate,
            aggregation_type,
            combination_type,
            normalize,
            name="graph_conv1",
        )
        # Create the second GraphConv layer.
        self.conv2 = GraphConvLayer(
            hidden_units,
            dropout_rate,
            aggregation_type,
            combination_type,
            normalize,
            name="graph_conv2",
        )
        # Create a postprocess layer.
        self.postprocess = create_ffn_last(hidden_units, dropout_rate, name="postprocess")
        # Create a compute logits layer.
        self.compute_logits = layers.Dense(units=num_classes, name="logits")
        self.node_embeddings_final = node_features

    def call(self, input_node_indices):
        # Preprocess the node_features to produce node representations.
        x = self.preprocess(self.node_features)
        # Apply the first graph conv layer.
        x1 = self.conv1((x, self.edges, self.edge_weights))
        # Skip connection.
        x = x1 + x
        # Apply the second graph conv layer.
        x2 = self.conv2((x, self.edges, self.edge_weights))
        # Skip connection.
        x = x2 + x
        # Postprocess node embedding.
        x = self.postprocess(x)
        self.node_embeddings_final= x
        # Fetch node embeddings for the input node_indices.
        node_embeddings = tf.gather(x, input_node_indices)
        # Compute logits
        return self.compute_logits(node_embeddings)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', help ='source folder')
    #parser.add_argument('--output_folder', help ='output folder')
    args = parser.parse_args()
    
    input_folder = args.input_folder
    #dataset =  pd.read_csv(os.path.join(input_folder,'dataset.csv'), dtype='object')
    dataset =  pd.read_pickle(os.path.join(input_folder,'dataset.pl'))
    dataset = dataset.sort_values('ids')
# Create an edges array (sparse adjacency matrix) of shape [2, num_edges].
   #edges, edge_weights = t
    edges = build_edge(dataset).T
    print('edges buided')
# Create an edge weights array of ones.
    #edge_weights = build_edge_weights(edges,dataset)
    edge_weights = tf.ones(shape=edges.shape[1])
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

    train_data, test_data = train_test_split(dataset, perc = 0.7)
    print("Train data shape:", train_data.shape)
    print("Test data shape:", test_data.shape)
    train_data.to_pickle(os.path.join
        (input_folder,'train_data.pl'))
    test_data.to_pickle(os.path.join
        (input_folder,'test_data.pl'))
    
    
   
    gnn_model = GNNNodeClassifier(
    graph_info=graph_info,
    num_classes=num_classes,
    hidden_units=hidden_units,
    conv_param=conv_param,
    pooling_param = pooling_param,
    dropout_rate=dropout_rate,
    name="gnn_model",
    )

    print("GNN output shape:", gnn_model([1, 10, 100]))
    gnn_model.summary()

    x_train = train_data.ids.to_numpy()
    y_train = train_data.label.to_numpy()
    history = run_experiment(gnn_model, x_train, y_train)
    gnn_model.save_weights(os.path.join(input_folder,"best_weights/Weights"))

    
    
    x_test = test_data.ids.to_numpy()
    y_test = test_data.label.to_numpy()
    _, test_accuracy = gnn_model.evaluate(x=x_test, y=y_test, verbose=0)
    print(f"Test accuracy: {round(test_accuracy * 100, 2)}%")
    _, train_accuracy = gnn_model.evaluate(x=x_train, y=y_train, verbose=0)
    print(f"Train accuracy: {round(train_accuracy * 100, 2)}%")
    # with open(os.path.join(input_folder,'node_repesentations.pkl'), 'wb') as f:  # open a text file
    #     pl.dumps(gnn_model.node_embeddings_final.eval())
