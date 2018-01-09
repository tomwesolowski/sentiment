import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf
import utils

from copy import deepcopy
from sklearn.manifold import TSNE
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tqdm import trange
from utils import SmartDict

# constants
num_classes = 2
review_size = 250
num_words = 400000
embedding_size = 50
epochs = 30
retries = 0
decay_step = 0.999
epochs_early_threshold = 10
accuracy_early_threshold = 0.55
swap_epochs_every = 3

valid_size = 2000
best_accuracy = 0.7

train_paths = SmartDict({
    'x': '../data/reviews_train_x.npy',
    'y': '../data/reviews_train_y.npy',
})
test_paths = SmartDict({
    'x': '../data/reviews_test_x.npy',
    'y': '../data/reviews_test_y.npy',
})

embeddings = np.load('../data/wordVectors.npy')
data = {
    'train': utils.Dataset(train_paths),
    'test': utils.Dataset(test_paths)
}

embeddings, num_words = utils.compress_embeddings(data, embeddings)
data['valid'] = data['train'].split_off(valid_size)

def input_fn(dataname, datasize, shuffle, params):
    with tf.variable_scope(dataname):
        # Dataset input
        x_placeholder = tf.placeholder(tf.int32, 
                                       shape=[datasize, review_size], 
                                       name='x_placeholder')
        y_placeholder = tf.placeholder(tf.int32, 
                                       shape=[datasize, num_classes], 
                                       name='y_placeholder')

        x_var = tf.Variable(
            x_placeholder, name="x_var", collections=[])
        y_var = tf.Variable(
            y_placeholder, name="y_var", collections=[])

        dataset = (tf.data.Dataset.from_tensor_slices(
            (x_var.initialized_value(),
             y_var.initialized_value()))
                .repeat()
                .prefetch(buffer_size=params.batch_size*2))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=params.batch_size)

        input_iterator = (dataset
            .apply(tf.contrib.data.batch_and_drop_remainder(params.batch_size))
            .make_initializable_iterator())             

        return SmartDict({
            "x": x_placeholder,
            "y": y_placeholder,
            "iterator": input_iterator,
            "initializer": input_iterator.initializer
        })


def build_model(params):    
    def _create_cell(model, id):
        with tf.variable_scope('cell_%d' % id, reuse=tf.AUTO_REUSE):
            cell_type = (
                tf.nn.rnn_cell.GRUCell if params.cell == 'GRUCell' else
                tf.nn.rnn_cell.LSTMCell
            )
            if cell_type is tf.nn.rnn_cell.GRUCell:
                cell = cell_type(params.state_size)
            else:
                cell = cell_type(params.state_size, state_is_tuple=True, use_peepholes=params.peep)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, 
                output_keep_prob=model.dropout)
            return cell
    
    def _create_network(dataset_it, name=None):
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE) as vs:
            model.set_prefix(name)

            model.dropout = tf.placeholder(tf.float32, shape=[])
            model.batch_inputs, model.batch_labels = dataset_it.get_next()

            model.embedded_batch_inputs = tf.nn.embedding_lookup(model['embeddings'], model.batch_inputs)

            model.cell = tf.nn.rnn_cell.MultiRNNCell(
                cells=[_create_cell(model, i) for i in range(params.layers)], 
                state_is_tuple=True)
            
            model.tuples_states = model.cell.zero_state(params.batch_size, dtype=tf.float32)
            
            model.states, model.current_state = tf.nn.dynamic_rnn(
                model.cell, model.embedded_batch_inputs, 
                initial_state=model.tuples_states,
                scope=vs)

            model.states = model.states[:, params.suffix:, :]

            model.Wproj = tf.tile(tf.expand_dims(
                tf.get_variable("Wproj", shape=(params.state_size, num_classes)), 0), 
                [params.batch_size, 1, 1])
            model.bproj = tf.tile(tf.expand_dims(
                tf.get_variable("bproj", shape=(1, num_classes)), 0), 
                [params.batch_size, 1, 1])
            model.logits = tf.matmul(model.states, model.Wproj) + model.bproj
            model.logits = tf.reduce_mean(model.logits, axis=1)

            model.predictions = tf.argmax(model.logits, axis=1)
            model.labels = tf.argmax(model.batch_labels, axis=1)
            model.correct_predictions = tf.reduce_sum(
                tf.cast(tf.equal(model.predictions, model.labels), tf.int32))

            model.loss = tf.nn.softmax_cross_entropy_with_logits(
                  logits=model.logits, labels=model.batch_labels)
            model.total_loss = tf.reduce_mean(model.loss)
            
            if params.decay:
                step = tf.get_variable("step", shape=[], initializer=tf.constant_initializer(0), trainable=False)
                model.lr = tf.train.exponential_decay(params.lr, step, 1, decay_step)
            else:
                model.lr = params.lr

            all_optimizers = {
                'GD': tf.train.GradientDescentOptimizer,
                'Adagrad': tf.train.AdagradOptimizer,
                'Adam': tf.train.AdamOptimizer,
            }

            model.optimizer = all_optimizers[params.optimizer](model.lr)

            if params.decay:
                model.opt = model.optimizer.minimize(model.total_loss, global_step=step)
            else:
                model.opt = model.optimizer.minimize(model.total_loss)

            tf.summary.scalar('correct_predictions', model.correct_predictions)
            model.all_summaries = tf.summary.merge_all()
            
            model.clear_prefix()
            
    tf.reset_default_graph()
    model = SmartDict()
    
    model.params = params

    model.params.name = model.name = params.name if params.name else utils.get_model_name(params)
    model.path = 'models/%s' % model.name
    print("Path: %s" % model.path)

    model.embeddings = tf.get_variable("embeddings", shape=[num_words, embedding_size])
    
    model.train_dataset = input_fn(
        'train', data['train'].num_examples, True, params)
    
    model.valid_dataset = input_fn(
        'valid', valid_size, False, params)
    
    _create_network(model.train_dataset.iterator)
    _create_network(model.valid_dataset.iterator, name='validation')
    
    model.writer = tf.summary.FileWriter(
        'models/%s/checkpoints/' % model.name, tf.get_default_graph(), flush_secs=10)
    model.saver = tf.train.Saver(
        var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    utils.describe_params(params)

    return model


def load_model(path, params):
    params = deepcopy(params)
    with open('%s/params.json' % path, 'r') as f:
        for name, value in json.load(f).items():
            if name not in params.values():
                params.add_hparam(name, value)
            else:
                params.set_hparam(name, value)
    return build_model(params)

def save_model(sess, model):
    model.saver.save(sess, '%s/model' % model.path)
    with open('%s/params.json' % model.path, 'w') as f:
        f.write(json.dumps(model.params.values()))


def validate(sess, model):
     # Validation
    num_batches = valid_size // model.params.batch_size
    tr_valid = trange(num_batches, leave=False)
    tr_valid.set_description('Validation')
    total = {}

    for batch_id in tr_valid:
        logits, labels, correct = sess.run(
            [model.validation_logits, model.validation_labels,
             model.validation_correct_predictions],
            feed_dict={
                model.validation_dropout: 1.0
        })
        probabilities = utils.softmax(logits)

        if 'probabilities' not in total:
            total['probabilities'] = probabilities
            total['labels'] = labels
            total['correct'] = 0.0
        else:
            total['probabilities'] = np.concatenate((total['probabilities'], probabilities))
            total['labels'] = np.concatenate((total['labels'], labels))
        total['correct'] += correct
    
    num_evaluated = total['probabilities'].shape[0]
    accuracy = total['correct'] / num_evaluated
    return accuracy, total['probabilities'], total['labels']
 

def train(paths, params, retries_left=retries):
    build_new_model = False
    if not paths:
        build_new_model = True
        paths.append('...') # hack

    for path in paths:
        model = build_model(params) if build_new_model else load_model(path, params)

        with utils.create_session() as sess:    
            vars_to_run = [
                model.train_dataset.initializer, model.valid_dataset.initializer
            ]
            if build_new_model:
                vars_to_run += [
                    tf.variables_initializer(
                        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))]
            else:
                model.saver.restore(sess, '%s/model' % model.path)

            sess.run(vars_to_run, 
                     feed_dict={
                         model.train_dataset.x: data['train'].x,
                         model.train_dataset.y: data['train'].y,
                         model.valid_dataset.x: data['valid'].x,
                         model.valid_dataset.y: data['valid'].y
            })

            average = {
                "loss": 1,
                "accuracy": 0.5,
                "num_correct": params.batch_size / 2
            }

            tr_epochs = trange(epochs)
            early_exitted = False

            # Training.
            for epoch in tr_epochs:
                num_batches = data['train'].num_examples // params.batch_size + 1
                tr_batch = trange(num_batches, leave=False)
                tr_batch.set_description('E %d/%d' % (epoch + 1, epochs))

                if (epoch+1) % swap_epochs_every == 0:
                    data['train'].random_swaps()

                for batch_id in tr_batch:
                    _, loss, num_correct, summaries, lr = sess.run(
                        [model.opt, model.total_loss, model.correct_predictions, model.all_summaries, model.lr],
                        feed_dict={
                            model.dropout: params.dropout
                        })

                    model.writer.add_summary(summaries, epoch * num_batches + batch_id)

                    average['loss'] = utils.move_average(average['loss'], loss)
                    average['num_correct'] = utils.move_average(average['num_correct'], num_correct)
                    tr_batch.set_postfix(loss=average['loss'], 
                                         accuracy=average['num_correct']/params.batch_size)
                    
                accuracy, _, _ = validate(sess, model)
                average['accuracy'] = utils.move_average(average['accuracy'], accuracy)

                global best_accuracy
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    save_model(sess, model)
                tr_epochs.set_postfix(lr=lr, best_accuracy=best_accuracy, validation_accuracy=accuracy)

                if epoch >= epochs_early_threshold and average['accuracy'] < accuracy_early_threshold:
                    early_exitted = True
                    print("Low avg. accuracy: %.3f after %d epochs. Stopping early. Retries left: %d" % (
                        average['accuracy'], epochs_early_threshold, retries_left))
                    break
            
        if early_exitted and retries_left > 0:
            return train(model_path, params, retries_left - 1)


def evaluate(paths, data, params):
    total = {}

    for path in paths:
        model = load_model(path, params)
        with utils.create_session() as sess:
            model.saver.restore(sess, '%s/model' % model.path)
        
            sess.run([model.valid_dataset.initializer], 
                     feed_dict={
                         model.valid_dataset.x: data.x,
                         model.valid_dataset.y: data.y
            })
         
            accuracy, probabilities, labels = validate(sess, model)
            if 'probabilities' not in total:
                total['probabilities'] = probabilities
                total['labels'] = labels
            else:
                total['probabilities'] += probabilities
            print("%s: " % model.name, accuracy)

    num_evaluated = total['probabilities'].shape[0]
    predictions = np.argmax(total['probabilities'], axis=1)
    num_correct = 0.0 + np.sum(predictions == total['labels'])
    accuracy = num_correct / num_evaluated
    print("Accuracy (%d models): %.3f" % (len(paths), accuracy))


def run(paths, dataset_name, params):
    if dataset_name == 'train':
        train(paths, params) 
    else:
        evaluate(paths, data[dataset_name], params)        


def run_all_configurations(params):
    def _inner_run(lst, params):
        if not lst:
            run([], 'train', params)
            return
        name, values = lst[0]
        for value in values:
            if name not in params.values():
                params.add_hparam(name, value)
            else:
                params.set_hparam(name, value)
            _inner_run(lst[1:], params)

    params = deepcopy(params)
    list_params = []
    for k, v in params.values().items():
        if type(v) is list:
            assert k[-4:] == '_all'
            list_params.append((k[:-4], v))
    _inner_run(list_params, params)


parser = argparse.ArgumentParser()
parser.add_argument('--hparams', type=str, default='')
parser.add_argument('--train', nargs='+', default=[])
parser.add_argument('--validate', nargs='+', default=[])
parser.add_argument('--test', nargs='+', default=[])
args = parser.parse_args()

params_def = tf.contrib.training.HParams(
    name='',
    batch_size_all=[valid_size // 3],
    state_size_all=[128],
    layers_all=[3],
    dropout_all=[0.85],
    lr_all=[0.005],
    optimizer_all=['Adam'],
    init_all=[None],
    peep_all=[True],
    decay_all=[True],
    cell_all=['LSTMCell'],
    suffix_all=[50, 150, 200]
)

if __name__ == '__main__':
    params_def.parse(args.hparams)
    provided_paths = (args.train or args.validate or args.test)
    if provided_paths:
        dataset_name = 'test' if args.test else 'valid' if args.validate else 'train'
        run(provided_paths, dataset_name, params_def)
    else:
        run_all_configurations(params_def) 