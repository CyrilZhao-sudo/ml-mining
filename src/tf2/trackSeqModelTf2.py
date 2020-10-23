# -*- coding: utf-8 -*-
# Author: zhao chen
# Date: 2020/10/14

import tensorflow as tf

from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, GlobalMaxPooling1D, concatenate, Layer
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers, initializers, constraints

from tensorflow.keras.metrics import AUC

import warnings
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc

warnings.filterwarnings("ignore")

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = tf.reshape(tf.matmul(tf.reshape(x, (-1, features_dim)),
                        tf.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = tf.tanh(eij)

        a = tf.exp(eij)

        if mask is not None:
            a *= tf.cast(mask, tf.float32)

        a /= tf.cast(tf.reduce_sum(a, axis=1, keepdims=True) + 1e-7, tf.float32)

        a = tf.expand_dims(a, axis=-1)
        weighted_input = x * a
        return tf.reduce_sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

    def get_config(self):
        config = {
            'step_dim': self.step_dim
        }
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ModelBase(object):

    def __init__(self, max_len, embed_size, vocab_size, metrics=None):
        self.max_len = max_len  # time steps
        self.embed_size = embed_size  # something like embed size
        self.vocab_size = vocab_size
        self.metrics = None if not metrics else metrics

        self.model = self.build_model()

    def build_model(self):
        raise NotImplementedError

    def fit(self, X_train, y_train, batch_size=36, epochs=2, valid_set=None, callbacks=None,class_weight=None):
        history = self.model.fit(X_train, y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 callbacks=callbacks,
                                 validation_data=valid_set, class_weight=class_weight
                                 )

        self.history = history.history

    def evaluate(self, X_valid, y_valid=None, tag="test", batch_size=512, verbose=0):
        y_prob = self.model.predict(X_valid, batch_size=batch_size, verbose=verbose)[:, 0]
        if y_valid is not None:
            fpr, tpr, _ = roc_curve(y_valid, y_prob)
            auc_score = auc(fpr, tpr)
            ks_score = max(abs(tpr - fpr))
            print("@@@ {} data auc: {:.4f}, ks: {:.4f}".format(tag, auc_score, ks_score))
        return y_prob

    def load_model(self, file_path, custom_objects=None):
        self.model = load_model(file_path, custom_objects=custom_objects)

    def load_weights(self, file_path):
        self.model.load_weights(file_path)

    def plot(self, save_path=None):
        import matplotlib.pyplot as plt
        plt.plot(self.history['loss'])
        plt.plot(self.history['val_loss'])
        plt.title("Model's Training & Validation loss across epochs")
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend(['Train', 'Valid'], loc='upper right')
        if save_path:
            plt.savefig(save_path)
        plt.show()

class LSTMBaseModel(ModelBase):
    def __init__(self, max_len, embed_size, vocab_size, metrics=None):
        super().__init__(max_len=max_len, embed_size=embed_size, vocab_size=vocab_size, metrics=metrics)

    def build_model(self):
        model = Sequential()
        model.add(Embedding(self.vocab_size, self.embed_size, input_length=self.max_len,
                            embeddings_regularizer=regularizers.l2(0.01)))
        # model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
        model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.5, recurrent_regularizer=regularizers.l2(0.01)))
        model.add(Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dense(1, activation="sigmoid"))
        print("LSTM Model summary : \n")
        model.summary()
        print("###" * 10)
        model.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=self.metrics)

        return model

    def __str__(self):
        return "LSTMBaseModel"


class AttentionBaseModel(ModelBase):
    def __init__(self, max_len, embed_size, vocab_size, metrics=None):
        super().__init__(max_len=max_len, embed_size=embed_size, vocab_size=vocab_size, metrics=metrics)

    def build_model(self):
        input = Input(shape=(self.max_len,))
        embed = Embedding(self.vocab_size, self.embed_size, input_length=self.max_len,
                          embeddings_regularizer=regularizers.l2(0.01))(input)
        lstm = LSTM(128, dropout=0.5, recurrent_dropout=0.5, recurrent_regularizer=regularizers.l2(0.01),
                    return_sequences=True)(embed)
        att_lstm = Attention(self.max_len)(lstm)
        max_pool_lstm = GlobalMaxPooling1D()(lstm)
        x = concatenate([att_lstm, max_pool_lstm])
        x = Dense(64, kernel_regularizer=regularizers.l2(0.01), activation="relu")(x)
        x = Dense(1, activation="sigmoid")(x)
        model = Model([input], x)

        print("Attention Model summary : \n")
        model.summary()
        print("###" * 10)
        model.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=self.metrics)

        return model

    def __str__(self):
        return "AttentionBaseModel"


def process_data(data, max_len, type='train', date_bound=None, label_name='label', seq_name='sequence', test_size=0.2,
                 drop_col=True,seed=2020):
    def get_seq(x):
        r = x.split(";")
        r = " ".join(r).strip() if len(r) < max_len else " ".join(r[-max_len:]).strip()
        return r

    data = data.copy()
    train_test_valid = {}
    if label_name != 'label':
        data['label'] = data[label_name]

    if type == 'train':
        data = data[~pd.isna(data['sequence'])]
        print("data columns: ", ", ".join(data.columns))
        doc_list = data[seq_name].str.split(';').tolist()
        data['feature'] = data[seq_name].apply(get_seq)
        if drop_col:
            data = data.drop(columns=[seq_name])
        if date_bound is None:
            train, test = train_test_split(data, test_size=test_size, random_state=seed)
            train_test_valid['train'] = train
            train_test_valid['test'] = test
            train_test_valid['valid'] = None
            train_test_valid['doc_list'] = doc_list

        else:
            train_test_valid['doc_list'] = doc_list

        return train_test_valid
    else:
        print("data columns: ", ", ".join(data.columns))
        data['feature'] = data[seq_name].apply(get_seq)
        if drop_col:
            data = data.drop(columns=[seq_name])
        train_test_valid['train'] = None
        train_test_valid['test'] = data
        train_test_valid['valid'] = None
        train_test_valid['doc_list'] = None
        return train_test_valid



def run(args):
    DATA_PATH = args.data_path
    MAX_LENGTH = args.max_length
    DATE_BOUND = args.date_bound
    EMBED_SIZE = args.embed_size
    BATCH_SIZE = args.batch_size
    PRED_BATCH_SIZE = args.pred_batch_size
    EPOCHS = args.epochs
    USE_GPU = args.use_gpu
    SEED = args.seed
    MODEL_PATH = args.model_path
    MODEL_TYPE = args.model_type
    MODE = args.mode

    if USE_GPU:
        # import os
        # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        import tensorflow as tf
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        # KTF.set_session(sess)

    if MODE == "train":

        data = pd.read_csv(DATA_PATH)

        train_test_valid = process_data(data, max_len=MAX_LENGTH, date_bound=DATE_BOUND, seed=SEED, test_size=0.3)
        doc_list = train_test_valid['doc_list']

        tokenizer = Tokenizer() # 记录所有的word index
        tokenizer.fit_on_texts(doc_list)
        word_index = tokenizer.word_index

        import os
        import pickle
        if not os.path.exists(MODEL_PATH + "/word_index"):
            with open(MODEL_PATH + "/word_index", 'wb') as f:
                pickle.dump(word_index, f)

        train_seq = tokenizer.texts_to_sequences(train_test_valid['train']["feature"])
        test_seq = tokenizer.texts_to_sequences(train_test_valid['test']["feature"])

        train_mat = pad_sequences(train_seq, maxlen=MAX_LENGTH)
        y_train = train_test_valid['train']['label'].values
        test_mat = pad_sequences(test_seq, maxlen=MAX_LENGTH)
        y_test = train_test_valid['test']['label'].values

        print("train size: {}, pos num: {}".format(len(train_mat), np.sum(y_train)))
        print("test size: {}, pos num: {}".format(len(test_mat), np.sum(y_test)))

        if MODEL_TYPE == "lstm":
            clf = LSTMBaseModel(max_len=MAX_LENGTH, vocab_size=len(word_index) + 1, embed_size=EMBED_SIZE, metrics=None)
        elif MODEL_TYPE == "attention":
            clf = AttentionBaseModel(max_len=MAX_LENGTH, vocab_size=len(word_index) + 1, embed_size=EMBED_SIZE,
                                     metrics=None)
            pass

        checkpoint = ModelCheckpoint("{0}/{1}.h5".format(MODEL_PATH, str(clf)), monitor='val_loss', verbose=1,
                                     mode='min', save_best_only=True)
        early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.01, restore_best_weights=True, patience=5)

        clf.fit(X_train=train_mat, y_train=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                valid_set=(test_mat, y_test), callbacks=[early_stopping, checkpoint], class_weight={0: 1, 1: 20})

        print("load_best_model ...")
        if MODEL_TYPE == "attention":
            clf.load_model("{0}/{1}.h5".format(MODEL_PATH, str(clf)), custom_objects={"Attention": Attention})
            pass
        else:
            clf.load_model("{0}/{1}.h5".format(MODEL_PATH, str(clf)))

        print("evaluate for train data ..")
        _ = clf.evaluate(X_valid=train_mat, y_valid=y_train, batch_size=PRED_BATCH_SIZE, tag='train')
        print("evaluate for test data ..")
        _ = clf.evaluate(X_valid=test_mat, y_valid=y_test, batch_size=PRED_BATCH_SIZE, tag='test')

        if DATE_BOUND:
            valid_seq = tokenizer.texts_to_sequences(train_test_valid['valid']["feature"])
            valid_mat = pad_sequences(valid_seq, maxlen=MAX_LENGTH)
            y_valid = train_test_valid['valid']['label'].values
            print("valid size: {}\n".format(len(valid_mat)))
            print("evaluate for test data ..")
            _ = clf.evaluate(X_valid=valid_mat, y_valid=y_valid, batch_size=PRED_BATCH_SIZE)

        clf.plot(save_path=MODEL_PATH + "/loss_fig.png")
    else:

        data = pd.read_csv(DATA_PATH)
        train_test_valid = process_data(data, type="test", max_len=MAX_LENGTH, date_bound=DATE_BOUND,seq_name="squence", seed=SEED, test_size=0.3)
        import os
        import pickle
        if os.path.exists(MODEL_PATH + "/word_index"):
            with open(MODEL_PATH + "/word_index", 'rb') as f:
                word_index = pickle.load(f)

        tokenizer = Tokenizer() #TODO 使用num_words导致最后一个word会缺失, 应该取消
        tokenizer.word_index = word_index

        test_seq = tokenizer.texts_to_sequences(train_test_valid['test']["feature"])
        test_mat = pad_sequences(test_seq, maxlen=MAX_LENGTH)

        if MODEL_TYPE == "lstm":
            clf = LSTMBaseModel(max_len=MAX_LENGTH, vocab_size=len(word_index) + 1, embed_size=EMBED_SIZE, metrics=None)
        elif MODEL_TYPE == "attention":
            clf = AttentionBaseModel(max_len=MAX_LENGTH, vocab_size=len(word_index) + 1, embed_size=EMBED_SIZE,
                                     metrics=None)
            pass

        print("load_best_model ...")
        if MODEL_TYPE == "attention":
            clf.load_model("{0}/{1}.h5".format(MODEL_PATH, str(clf)), custom_objects={"Attention": Attention})
            pass
        else:
            clf.load_model("{0}/{1}.h5".format(MODEL_PATH, str(clf)))
        print("predict the test ...")
        y_prob = clf.evaluate(test_mat, batch_size=PRED_BATCH_SIZE)

        sub = train_test_valid['test'][["uid"]]
        sub["score"] = y_prob
        sub.to_csv(MODEL_PATH + "/sub_{}.txt".format(str(clf)),index=False, header=None)



if __name__ == "__main__":
    import argparse
    import time

    start = time.time()

    parser = argparse.ArgumentParser(description="behavior sequence model script")
    parser.add_argument("--data_path", type=str, default="/home/mi/data/sequence/target_ids_c.csv")
    parser.add_argument("--max_length", type=int, default=36) # 15 36
    parser.add_argument("--date_bound", type=int, default=None)
    parser.add_argument("--embed_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--use_gpu", type=bool, default=False)
    parser.add_argument("--pred_batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--model_path", type=str, default="/home/mi/PycharmProjects/BehaviorSequence/resources")
    parser.add_argument("--model_type", type=str, default="attention")
    parser.add_argument("--mode", type=str, default="test")
    args = parser.parse_args()

    run(args)

    end = time.time()
    print("train model used time: {} m".format(round((end - start) / 60), 2))
