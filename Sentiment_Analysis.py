import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import tensor
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator
import data_loader
import pickle
import tqdm

# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"

dataset = data_loader.SentimentTreeBank()

# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.vocab.keys())
    print(wv_from_bin.vocab[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=True):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    embeddings = [word_to_vec.get(word) for word in sent.text]
    embeddings = [x for x in embeddings if x is not None]
    if not embeddings:
        return np.zeros(300)
    embeddings = np.stack(embeddings, axis=0)
    return np.mean(embeddings, axis=0)


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    one_hot = np.zeros(np.int32(size))
    one_hot[ind] = 1
    return one_hot


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    average_vector = np.zeros(len(word_to_ind))
    for word in sent.text:
        index = word_to_ind.get(word)
        average_vector[index] += 1
    return average_vector/len(sent.text)


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    index = list(np.arange(len(words_list)))
    return dict(zip(words_list,index))


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """
    sent_embedding = np.zeros((seq_len, embedding_dim))
    sentence = sent.text
    if len(sentence) >= seq_len:
        for i in range(seq_len):
            embedding = word_to_vec.get(sentence[i])
            if embedding is not None:
                sent_embedding[i,:] = embedding
    else:
        for i in range(len(sentence)):
            embedding = word_to_vec.get(sentence[i])
            if embedding is not None:
                sent_embedding[i,:] = embedding
    return sent_embedding


class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank", batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs) for
                               k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape




# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """
    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,num_layers=n_layers,bidirectional=True)
        self.linear = nn.Linear(in_features=2*hidden_dim, out_features=1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, text):
        return self.linear(self.dropout(torch.cat(self.lstm(text.permute(1, 0, 2))[1][0].unbind(0), 1)))

    def predict(self, text):
        return


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """
    def __init__(self, embedding_dim):
        super().__init__()
        self.linear = nn.Linear(in_features=embedding_dim, out_features=1)

    def forward(self, x):
        return self.linear(x)

    def predict(self, x):
        return torch.round(nn.Sigmoid()(self.linear(x.float())))



# ------------------------- training functions -------------


def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    return np.mean((preds>0.5) == y)


def train_epoch(model, data_iterator, optimizer, criterion):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """
    for input_batch, label_batch in tqdm.tqdm(data_iterator):
        optimizer.zero_grad()
        pred = model.forward(input_batch.float())
        loss = criterion(pred, torch.unsqueeze(label_batch.float(), 1))
        loss.backward()
        optimizer.step()


def evaluate(model, data_iterator, criterion):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """
    preds, true_labels = get_predictions_for_data(model, data_iterator)
    array_of_loss = criterion(torch.tensor(preds), torch.tensor(true_labels))
    return tuple([np.mean(np.array(array_of_loss)),binary_accuracy(np.array(nn.Sigmoid()((torch.tensor(preds)))),np.array(true_labels))])

def get_predictions_for_data(model, data_iter):
    """

    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return:
    """
    preds = np.array([])
    true_labels = np.array([])
    for input_batch, label_batch in tqdm.tqdm(data_iter):
        preds = np.append(preds, model.forward(input_batch.float()).detach().numpy())
        true_labels = np.append(true_labels, torch.unsqueeze(label_batch.float(),1).numpy())
    return preds, true_labels


def train_model(model, data_manager, n_epochs, lr, model_type, weight_decay=0):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    train_loss = np.zeros(n_epochs+1)
    train_accuracy = np.zeros(n_epochs+1)
    train_loss[0], train_accuracy[0] = np.array(evaluate(model, data_manager.get_torch_iterator(data_subset=TRAIN), criterion))
    validation_loss = np.zeros(n_epochs+1)
    validation_accuracy = np.zeros(n_epochs+1)
    validation_loss[0], validation_accuracy[0] = np.array(evaluate(model, data_manager.get_torch_iterator(data_subset=VAL), criterion))
    for i in range(n_epochs):
        optimizer.zero_grad()
        train_epoch(model, data_manager.get_torch_iterator(data_subset=TRAIN), optimizer, criterion)
        train_loss[i+1], train_accuracy[i+1] = np.array(evaluate(model, data_manager.get_torch_iterator(data_subset=TRAIN), criterion))
        validation_loss[i+1], validation_accuracy[i+1] = np.array(evaluate(model, data_manager.get_torch_iterator(data_subset=VAL), criterion))
    epoch_round = np.arange(0,n_epochs+1)
    plt.subplot(1,2,1)
    plt.plot(epoch_round, train_loss, label='Train')
    plt.plot(epoch_round, validation_loss, label='Validation')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Cross entropy loss')
    plt.title(model_type+' loss:')
    plt.subplot(1,2,2)
    plt.plot(epoch_round, train_accuracy, label='Train')
    plt.plot(epoch_round, validation_accuracy, label='Validation')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(model_type+' accuracy:')
    plt.show()
    test_sentences = dataset.get_test_set()
    test_accuracy = evaluate(model, data_manager.get_torch_iterator(data_subset=TEST), criterion)[1]
    negated_polarity_test_subset = data_loader.get_negated_polarity_examples(test_sentences)
    negated_polarity_indices = data_loader.get_negated_polarity_examples(test_sentences)
    negated_polarity_subset = [test_sentences[i] for i in negated_polarity_indices]
    rare_words_indices = data_loader.get_rare_words_examples(test_sentences, dataset)
    rare_words_subset = [test_sentences[i] for i in rare_words_indices]
    negated_polarity_dataset = OnlineDataset(negated_polarity_subset, data_manager.sent_func, data_manager.sent_func_kwargs)
    rare_words_dataset = OnlineDataset(rare_words_subset, data_manager.sent_func, data_manager.sent_func_kwargs)
    negated_polarity_accuracy = evaluate(model,DataLoader(negated_polarity_dataset,64), criterion=criterion)[1]
    rare_words_accuracy = evaluate(model,DataLoader(rare_words_dataset,64), criterion=criterion)[1]
    print('With',model_type,':\nTest accuracy: ',test_accuracy,'\nNegated Polarity accuracy: ',negated_polarity_accuracy)
    print('Rare Words accuracy: ', rare_words_accuracy)


def train_log_linear_with_one_hot():
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """
    data_manager = DataManager(data_type=ONEHOT_AVERAGE, batch_size=64)

    model = LogLinear(data_manager.get_input_shape()[0])
    train_model(model, data_manager, n_epochs=20, lr=0.01, weight_decay=0.0001, model_type=ONEHOT_AVERAGE)



def train_log_linear_with_w2v():
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """
    data_manager = DataManager(data_type=W2V_AVERAGE, batch_size=64, embedding_dim=300)
    model = LogLinear(data_manager.get_input_shape()[0])
    train_model(model, data_manager, n_epochs=20, lr=0.01, weight_decay=0.0001, model_type=W2V_AVERAGE)


def train_lstm_with_w2v():
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    data_manager = DataManager(data_type=W2V_SEQUENCE, batch_size=64,embedding_dim=300)
    model = LSTM(embedding_dim=300,hidden_dim=100,n_layers=1,dropout=0.5)
    train_model(model=model,data_manager=data_manager,n_epochs=4,lr=0.001, weight_decay=0.0001,model_type=W2V_SEQUENCE)


if __name__ == '__main__':
    train_log_linear_with_one_hot()
    train_log_linear_with_w2v()
    train_lstm_with_w2v()


