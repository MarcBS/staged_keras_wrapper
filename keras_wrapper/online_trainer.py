# -*- coding: utf-8 -*-
import copy
import logging

import numpy as np

from keras_wrapper.cnn_model import Model_Wrapper
from keras_wrapper.extra.read_write import list2file
from keras_wrapper.utils import indices_2_one_hot, decode_predictions_beam_search


class OnlineTrainer:
    def __init__(self, models, dataset, sampler, params_prediction=None, params_training=None, verbose=0):
        """
        Class for performing online training in a post-editing scenario.

        :param models: Models to use for decoding/training.
        :param dataset: Dataset instance for the task at hand.
        :param sampler: Sampler instance (e.g. BeamSearcher) for decoding
        :param params_prediction: Parameters for making predictions
        :param params_training: Training parameters
        :param verbose: Be verbose or not
        """
        self.models = models
        self.dataset = dataset
        self.sampler = sampler
        self.verbose = verbose
        self.params_prediction = self.checkParameters(params_prediction) if params_prediction is not None else {}
        self.params_training = self.checkParameters(params_training, params_training=True) if params_training \
                                                                                              is not None else None
        self.sentence_scorer = None
        target_vocabulary_id = params_prediction['dataset_outputs'][
            0] if params_prediction is not None else 'target_text'
        self.index2word_y = self.dataset.vocabulary[target_vocabulary_id]['idx2words']
        self.mapping = None if self.dataset.mapping == dict() else self.dataset.mapping
        self.n_updates = 0
        if self.params_prediction.get('n_best_optimizer', False):
            self.words2index_y = self.dataset.vocabulary[target_vocabulary_id]['words2idx']
        if self.params_prediction.get('optimizer_regularizer', 'ter').lower() == 'ter':
            from pycocoevalcap.sentence_ter.sentence_ter import SentenceTerScorer
            self.sentence_scorer = SentenceTerScorer('')

        elif self.params_prediction.get('optimizer_regularizer').lower() == 'bleu':
            from pycocoevalcap.sentence_bleu.sentence_bleu import SentenceBleuScorer
            self.sentence_scorer = SentenceBleuScorer('')

    def sample_and_train_online(self, X, Y, src_words=None, trg_words=None):
        """
        Online training in a post-editing scenario.
        First, we make a sample with the model(s).
        Next, we post-edit it (using the reference).
        Finally, we update our model(s).
        :param X: Model inputs (source_text, state_below)
        :param Y: Model outputs (target text)
        :param src_words: Sequence of source words
        :param trg_words: Sequence of target words
        :return: None
        """
        # Get source and target samples (post-edited sentences - references)
        x = X[0]
        state_below_y = X[1]
        y = Y[0]

        # If necessary, set reference in scorer
        if self.params_prediction.get('optimizer_regularizer').lower() == 'bleu' or \
                        self.params_prediction.get('optimizer_regularizer').lower() == 'ter':
            self.sentence_scorer.set_reference(str(trg_words[0]).split())

        # 1. Translate source sentence with the current model. Get a single hypothesis or the N-best list
        if self.params_prediction['n_best_optimizer']:
            [trans_indices, costs, alphas], n_best = self.sampler.sample_beam_search(x[0])

        else:
            trans_indices, costs, alphas = self.sampler.sample_beam_search(x[0])

        # Decode best hypothesis
        if self.params_prediction['pos_unk']:
            alphas = [alphas]
            sources = [x] if not src_words else src_words
            heuristic = self.params_prediction['heuristic']
        else:
            alphas = None
            heuristic = None
            sources = None

        if self.params_prediction['store_hypotheses'] is not None or self.sentence_scorer is not None:
            hypothesis = decode_predictions_beam_search([trans_indices],
                                                        self.index2word_y,
                                                        alphas=alphas,
                                                        x_text=sources,
                                                        heuristic=heuristic,
                                                        mapping=self.mapping,
                                                        pad_sequences=True,
                                                        verbose=0)[0]
            # Apply detokenization function if needed
            if self.params_prediction.get('apply_detokenization', False):
                hypothesis_to_write = self.params_prediction['detokenize_f'](hypothesis)
            else:
                hypothesis_to_write = hypothesis
            # Store original hypothesis
            if self.params_prediction['store_hypotheses'] is not None:
                list2file(self.params_prediction['store_hypotheses'], [hypothesis_to_write + '\n'], permission='a')

            if self.verbose > 1:
                logging.info('Hypothesis: %s' % str(hypothesis_to_write))

        # If we are working with an n-best list, we'll probably have to decode it
        if self.params_prediction['n_best_optimizer']:
            if self.verbose > 0:
                print ""
                print "\tReference: ", trg_words[0]

            # Get max length of the hypotheses in the N-best list, for a future mini-batch training.
            maxlen_nbest_hypothesis = y.shape[1] + 1
            # Decode N-best list
            for n_best_preds, n_best_scores, n_best_alphas in n_best:
                n_best_predictions = []

                for i, (n_best_pred, n_best_score, n_best_alpha) in enumerate(zip(n_best_preds,
                                                                                  n_best_scores,
                                                                                  n_best_alphas)):
                    if len(n_best_pred) + 1 > maxlen_nbest_hypothesis:
                        maxlen_nbest_hypothesis = len(n_best_pred) + 1
                    pred = decode_predictions_beam_search([n_best_pred],
                                                          self.index2word_y,
                                                          alphas=[n_best_alpha],
                                                          x_text=sources,
                                                          heuristic=heuristic,
                                                          mapping=self.mapping,
                                                          pad_sequences=True,
                                                          verbose=0)
                    # Apply detokenization function if needed
                    if self.params_prediction.get('apply_detokenization', False):
                        pred = map(self.params_prediction['detokenize_f'], pred)

                    # Score N_best scores
                    hypothesis_to_score = pred[0].split()

                    if self.sentence_scorer is not None:
                        if self.params_prediction.get('optimizer_regularizer', 'ter').lower() == 'ter':
                            score = self.sentence_scorer.score(hypothesis_to_score)
                        elif self.params_prediction.get('optimizer_regularizer').lower() == 'bleu':
                            # We are always minimizing, therefore, we use 1 - BLEU as score.
                            score = 1. - self.sentence_scorer.score(hypothesis_to_score)
                    else:
                        score = n_best_score
                    n_best_predictions.append([i, n_best_pred, pred, score])

            # Add reference to the bottom of the N-best list ?
            # n_best_predictions.append([i + 1, one_hot_2_indices(y)[0], trg_words, 0.0])

            # Sort n-best list according to the metric
            p = np.argsort([nbest[3] for nbest in n_best_predictions])
            # n_best_predictions format:
            # [[id_1, [trans_indices_1], [trans_words_1], score_1],
            #  [id_2, [trans_indices_2], [trans_words_2], score_2],
            #   ...
            # [id_N, [trans_indices_N], [trans_words_N], score_N]]
            # p: argsort of scores from n_best_predictions

            train_inputs = []
            train_outputs = []
            # Our goal is to sort the Nbest list according to the MT metric, instead of logprob.
            # We want to modify the model in a way that it gives more logprob to those samples with better metric
            # That means, to make that top-metric hypotheses are located in top-positions of the N-best list.
            for i in range(len(p)):
                for j in range(i, len(p)):
                    if i != j:
                        # TODO: Document this code
                        permutation_index_low = p[i]
                        permutation_index_high = p[j]
                        if permutation_index_low <= permutation_index_high:
                            print "The %d-th top-prob hypothesis is ok with respect to the %d-th" \
                                  " top-metric hypothesis " % (i, j,)
                            print "----------"
                        else:
                            if self.verbose:
                                print "Mismatch in positions %d and %d. The %d-th hypothesis should be the %d" % \
                                      (i, j, permutation_index_low, permutation_index_high)
                                print "top-prob h:", n_best_predictions[permutation_index_high][2][0], \
                                    self.params_prediction.get('optimizer_regularizer') + ":", \
                                    n_best_predictions[permutation_index_high][3]
                                print "top_metric_h", n_best_predictions[permutation_index_low][2][0], \
                                    self.params_prediction.get('optimizer_regularizer') + ":", \
                                    n_best_predictions[permutation_index_low][3]

                            # Log diff loss: p(h_i|x) - p(h_j|x) -> We need to compute 2 logprobs
                            if 'log_diff' in self.params_training.get('loss').keys()[0]:
                                # Tensors for computing p(h_i|x)
                                top_metric_h = np.zeros(maxlen_nbest_hypothesis, dtype='int64')
                                unnormalized_top_metric_h = np.asarray(n_best_predictions[permutation_index_low][1])
                                top_metric_h[:len(unnormalized_top_metric_h)] = unnormalized_top_metric_h
                                state_below_top_metric_h = np.zeros(maxlen_nbest_hypothesis, dtype='int64')
                                unnormalized_state_below_top_metric_h = \
                                    np.asarray(np.append(self.dataset.extra_words['<null>'], top_metric_h[:-1]))
                                state_below_top_metric_h[:len(unnormalized_state_below_top_metric_h)] = \
                                    unnormalized_state_below_top_metric_h
                                top_metric_h = np.array([indices_2_one_hot(top_metric_h,
                                                                           self.dataset.vocabulary_len["target_text"])])
                                # Tensors for computing p(h_j|x)
                                unnormalized_top_prob_h = np.asarray(n_best_predictions[permutation_index_high][1])
                                top_prob_h = np.zeros(maxlen_nbest_hypothesis, dtype='int64')
                                top_prob_h[:len(unnormalized_top_prob_h)] = unnormalized_top_prob_h
                                state_below_top_prob_h = np.zeros(maxlen_nbest_hypothesis, dtype='int64')
                                unnormalized_state_below_top_prob_h = \
                                    np.asarray(np.append(self.dataset.extra_words['<null>'], top_prob_h[:-1]))
                                state_below_top_prob_h[:len(unnormalized_state_below_top_prob_h)] = \
                                    unnormalized_state_below_top_prob_h
                                top_prob_h = np.array([indices_2_one_hot(top_prob_h,
                                                                         self.dataset.vocabulary_len["target_text"])])
                                # Build model inputs
                                if 'log_diff_plus_categorical_crossentropy' in self.params_training.get('loss').keys()[
                                    0]:
                                    current_permutation_train_inputs = [x, state_below_y,
                                                                        np.asarray([state_below_top_metric_h]),
                                                                        np.asarray([state_below_top_prob_h]),
                                                                        np.asarray([self.params_training[
                                                                            'additional_training_settings'].get(
                                                                            'lambda', 0.5)])] + \
                                                                       [y, top_metric_h, top_prob_h]
                                else:
                                    current_permutation_train_inputs = [x, state_below_top_metric_h,
                                                                        state_below_top_prob_h] + [top_metric_h,
                                                                                                   top_prob_h]
                                # Add inputs to current minibatch
                                if train_inputs:
                                    for input_index in range(len(current_permutation_train_inputs)):
                                        train_inputs[input_index] = \
                                            np.vstack((train_inputs[input_index],
                                                       current_permutation_train_inputs[input_index]))
                                else:
                                    train_inputs = current_permutation_train_inputs
                                # Dummy outputs for our dummy loss
                                train_outputs = np.zeros((train_inputs[0].shape[0], 1), dtype='float32')
                            elif 'signed_categorical_crossentropy' in self.params_training.get('loss').keys()[0]:
                                unnormalized_top_prob_h = np.asarray(n_best_predictions[permutation_index_high][1])
                                top_prob_h = np.zeros(maxlen_nbest_hypothesis, dtype='int64')
                                top_prob_h[:len(unnormalized_top_prob_h)] = unnormalized_top_prob_h
                                state_below_top_prob_h = np.zeros(maxlen_nbest_hypothesis, dtype='int64')
                                unnormalized_state_below_top_prob_h = \
                                    np.asarray(np.append(self.dataset.extra_words['<null>'], top_prob_h[:-1]))
                                state_below_top_prob_h[:len(unnormalized_state_below_top_prob_h)] = \
                                    unnormalized_state_below_top_prob_h
                                top_prob_h = np.array([indices_2_one_hot(top_prob_h,
                                                                         self.dataset.vocabulary_len["target_text"])])
                                sign = np.array([self.params_training['additional_training_settings'].get('tau', 1)
                                                 * n_best_predictions[permutation_index_low][3]])
                                current_permutation_train_inputs = [x, state_below_top_prob_h, top_prob_h, sign]
                                current_permutation_train_outputs = []

                                if train_inputs:
                                    for input_index in range(len(current_permutation_train_inputs)):
                                        train_inputs[input_index] = np.vstack((train_inputs[input_index],
                                                                               current_permutation_train_inputs[
                                                                                   input_index]))
                                else:
                                    train_inputs = [x, state_below_top_prob_h, top_prob_h, sign]
                                train_outputs = np.zeros((train_inputs[0].shape[0], 1), dtype='float32')
                            else:
                                raise NotImplementedError, 'The loss function " %s " is still unimplemented.' % \
                                                           self.params_training.get('loss')
            # Update the models
            for model in self.models:
                # We are always working with Model from Keras
                if isinstance(model, Model_Wrapper):
                    model = model.model
                if not train_inputs:
                    # Use references
                    train_inputs = [x, state_below_y, state_below_y, state_below_y,
                                    np.asarray([1.]),
                                    y, y, y]
                    train_outputs = np.zeros((train_inputs[0].shape[0], 1), dtype='float32')
                if train_inputs:
                    # The PAS algorithm requires to set weights and switch the loss subderivative to 1. or 0.
                    if 'pas' in self.params_training['optimizer']:
                        weights = model.trainable_weights
                        # Weights from Keras 2 are already (topologically) sorted!
                        # weights.sort(key=lambda x: x.name if x.name else x.auto_name)
                        model.optimizer.set_weights(weights)
                        model.optimizer.loss_value.set_value(1.0)
                    # We may want to iterate over the same sample over and over
                    for k in range(self.params_training['additional_training_settings'].get('k', 1)):
                        model.fit(x=train_inputs,
                                  y=train_outputs,
                                  batch_size=min(self.params_training['batch_size'], train_inputs[0].shape[0]),
                                  epochs=self.params_training['n_epochs'],
                                  verbose=self.params_training['verbose'],
                                  callbacks=[],
                                  validation_data=None,
                                  validation_split=self.params_training.get('val_split', 0.),
                                  shuffle=self.params_training['shuffle'],
                                  class_weight=None,
                                  sample_weight=None,
                                  initial_epoch=0)
                        self.n_updates += 1
                    del train_inputs

        else:  # Not N-Best-based optimizer
            # 2. Post-edit this sample in order to match the reference --> Use y
            # 3. Update net parameters with the corrected samples
            for model in self.models:
                if self.params_training.get('use_custom_loss', False):
                    loss = 1
                    # With custom losses, we'll probably use the hypothesis as training sample -> Convert to one-hot
                    # Tensors for computing p(h_i|x)
                    hyp = np.array([indices_2_one_hot(trans_indices, self.dataset.vocabulary_len["target_text"])])
                    state_below_h = np.asarray([np.append(self.dataset.extra_words['<null>'], trans_indices[:-1])])
                    # Build model inputs according to those required for each loss function
                    if 'log_diff' in self.params_training.get('loss'):
                        train_inputs = [x, state_below_y, state_below_h] + [y, hyp]
                    elif 'log_diff_plus_categorical_crossentropy' in self.params_training.get('loss').keys()[0]:
                        train_inputs = [x, state_below_y, state_below_y, state_below_h,
                                        np.asarray([self.params_training['additional_training_settings'].get('lambda', 0.5)])]  +\
                                       [y, y, hyp]
                    elif 'weighted_log_diff' in self.params_training.get('loss').keys()[0]:
                        train_inputs = [x, state_below_y, state_below_h,
                                        np.asarray([self.params_training['additional_training_settings'].get('lambda', 0.5)])]  +\
                                       [y, hyp]

                    elif 'hybrid_log_diff' in self.params_training.get('loss').keys()[0]:
                        train_inputs = [x, state_below_y, state_below_h, y, hyp,
                                        np.asarray([1.]), np.asarray([0.]), np.asarray([0.])]
                        p_t_y = model.predict(train_inputs, batch_size=1, verbose=0)
                        train_inputs = [x, state_below_y, state_below_h, y, hyp,
                                        np.asarray([self.params_training['additional_training_settings'].get('d', 0.5)]),
                                        np.asarray([self.params_training['additional_training_settings'].get('c', 0.5)]),
                                        p_t_y]

                    # Dummy outputs for our dummy loss
                    train_outputs = np.zeros((train_inputs[0].shape[0], 1), dtype='float32')
                    # The PAS-like algorithms require to set weights and switch the loss subderivative  to 1. or 0.
                    if 'pas' in self.params_training['optimizer']:
                        weights = model.trainable_weights
                        # Weights from Keras 2 are already (topologically) sorted!
                        # weights.sort(key=lambda x: x.name if x.name else x.auto_name)
                        model.optimizer.set_weights(weights)
                        # The PAS algorithm requires to set weights
                        # and switch the loss subderivative  to 1. or 0.
                        loss_val = model.evaluate(train_inputs,
                                                  np.zeros((y.shape[0], 1), dtype='float32'),
                                                  batch_size=1, verbose=0)
                        loss = 1.0 if loss_val > 0 else 0.0
                        model.optimizer.loss_value.set_value(loss)

                    for k in range(self.params_training['additional_training_settings'].get('k', 1)):
                        # Fit!
                        model.fit(x=train_inputs,
                                  y=train_outputs,
                                  batch_size=min(self.params_training['batch_size'], len(x)),
                                  epochs=self.params_training['n_epochs'],
                                  verbose=self.params_training['verbose'],
                                  callbacks=[],
                                  validation_data=None,
                                  validation_split=self.params_training.get('val_split', 0.),
                                  shuffle=self.params_training['shuffle'],
                                  class_weight=None,
                                  sample_weight=None,
                                  initial_epoch=0)
                        self.n_updates += loss
                    # We are optimizing towards an MT metric (BLEU or TER)
                    if self.params_prediction.get('optimizer_regularizer').lower() == 'bleu' or \
                                    self.params_prediction.get('optimizer_regularizer').lower() == 'ter':
                        # Get score of the hypothesis (after post-processing)
                        hypothesis_to_score = hypothesis_to_write.split()
                        if self.sentence_scorer is not None:
                            if self.params_prediction.get('optimizer_regularizer', 'ter').lower() == 'ter':
                                score = self.sentence_scorer.score(hypothesis_to_score)
                            elif self.params_prediction.get('optimizer_regularizer').lower() == 'bleu':
                                # We are always minimizing, therefore, we use 1 - BLEU as score.
                                score = 1. - self.sentence_scorer.score(hypothesis_to_score)
                        # Build the training inputs
                        train_inputs = [x, state_below_y, y,
                                        np.array([max(1e-8, 1. - score)]),
                                        np.array([self.params_training['additional_training_settings'].get('lambda',
                                                                                                           0.5)])]

                        train_outputs = np.zeros((y.shape[0], 1), dtype='float32')
                        # Fit!
                        model.fit(x=train_inputs,
                                  y=train_outputs,
                                  batch_size=min(self.params_training['batch_size'], train_inputs[0].shape[0]),
                                  epochs=self.params_training['n_epochs'],
                                  verbose=self.params_training['verbose'],
                                  callbacks=[],
                                  validation_data=None,
                                  validation_split=self.params_training.get('val_split', 0.),
                                  shuffle=self.params_training['shuffle'],
                                  class_weight=None,
                                  sample_weight=None,
                                  initial_epoch=0)
                        self.n_updates += 1
                else:
                    # Classical setup: We can train the TranslationModel directly
                    # We can include MT metrics as a LR modifier
                    if self.params_prediction.get('optimizer_regularizer').lower() == 'bleu' or \
                                    self.params_prediction.get('optimizer_regularizer').lower() == 'ter':
                        # Get score of the hypothesis (after post-processing)
                        hypothesis_to_score = hypothesis_to_write.split()
                        if self.sentence_scorer is not None:
                            if self.params_prediction.get('optimizer_regularizer', 'ter').lower() == 'ter':
                                score = self.sentence_scorer.score(hypothesis_to_score)
                            elif self.params_prediction.get('optimizer_regularizer').lower() == 'bleu':
                                # We are always minimizing, therefore, we use 1 - BLEU as score.
                                score = 1. - self.sentence_scorer.score(hypothesis_to_score)

                        # Adjust LR for the current sample according to its value of TER/BLEU
                        model.model.optimizer.set_lr(self.params_training['lr'] *
                                                     self.params_training['additional_training_settings'].get('lambda',
                                                                                                              0.5) *
                                                     score)

                    params = copy.copy(self.params_training)
                    # Remove unnecessary parameters
                    del params['use_custom_loss']
                    del params['n_best_optimizer']
                    del params['optimizer']
                    del params['lr']
                    del params['additional_training_settings']
                    del params['loss']
                    # Train!
                    model.trainNetFromSamples([x, state_below_y], y, params)
                    self.n_updates += 1

    def train_online(self, X, Y, trans_indices=None, trg_words=None, n_best=None):
        """
        Online training in a post-editing scenario. We only apply the training step.
        :param X: Model inputs (source_text, state_below)
        :param Y: Model outputs (target text)
        :param trans_indices: Indices of words for the prediction made by the model
        :param trg_words: Sequence of target words
        :param n_best: n_best list (in case of n-best-based optimizers)
        :return: None
        """
        x = X[0]
        state_below_y = X[1]
        y = Y[0]

        if self.params_training.get('use_custom_loss', False):
            raise NotImplementedError, 'Custom loss + train_online (interactive)' \
                                       ' still unimplemented. Refer to sample_and_train_online'
            state_below_h = np.asarray([np.append(self.dataset.extra_words['<null>'], trans_indices[:-1])])
            hyp = np.array([indices_2_one_hot(trans_indices, self.dataset.vocabulary_len["target_text"])])

        if self.params_prediction.get('n_best_optimizer', False):
            raise Exception, 'N-best optimizer + train_online (interactive)' \
                             ' still unimplemented. Refer to sample_and_train_online'
            if self.verbose > 0:
                print ""
                print "\tReference: ", trg_words[0]
            for n_best_preds, n_best_scores, n_best_alphas in n_best:
                n_best_predictions = []
                for i, (n_best_pred, n_best_score, n_best_alpha) in enumerate(zip(n_best_preds,
                                                                                  n_best_scores,
                                                                                  n_best_alphas)):
                    pred = decode_predictions_beam_search([n_best_pred],
                                                          self.index2word_y,
                                                          alphas=[n_best_alpha],
                                                          x_text=sources,
                                                          heuristic=heuristic,
                                                          mapping=self.mapping,
                                                          pad_sequences=True,
                                                          verbose=0)
                    # Apply detokenization function if needed
                    if self.params_prediction.get('apply_detokenization', False):
                        pred = map(self.params_prediction['detokenize_f'], pred)

                    if self.sentence_scorer is not None:
                        # We are always minimizing, therefore, we use 1 - BLEU as score.
                        score = 1. - self.sentence_scorer.score(pred[0].split())
                    else:
                        score = n_best_score
                    n_best_predictions.append([i, n_best_pred, pred, score])
            for model in self.models:
                weights = model.trainable_weights
                # weights.sort(key=lambda x: x.name if x.name else x.auto_name)
                model.optimizer.set_weights(weights)
                top_prob_h = np.asarray(n_best_predictions[0][1])
                p = np.argsort([nbest[3] for nbest in n_best_predictions])
                if p[0] == 0:
                    if self.verbose > 0:
                        print "The top-prob hypothesis and the top bleu hypothesis match"
                else:
                    if self.verbose:
                        print "The top-prob hypothesis and the top bleu hypothesis don't match"
                        print "top-prob h:", n_best_predictions[0][2][0], "Bleu:", 1 - n_best_predictions[0][-1]
                        print "top_bleu_h", n_best_predictions[p[0]][2][0], "Bleu:", 1 - n_best_predictions[p[0]][-1]
                        print "Updating..."

                    top_bleu_h = np.asarray(n_best_predictions[p[0]][1])
                    state_below_top_prob_h = np.asarray([np.append(self.dataset.extra_words['<null>'],
                                                                   top_prob_h[:-1])])
                    state_below_top_bleu_h = np.asarray([np.append(self.dataset.extra_words['<null>'],
                                                                   top_bleu_h[:-1])])
                    top_prob_h = np.array([indices_2_one_hot(top_prob_h,
                                                             self.dataset.vocabulary_len["target_text"])])
                    top_bleu_h = np.array([indices_2_one_hot(top_bleu_h,
                                                             self.dataset.vocabulary_len["target_text"])])

                    train_inputs = [x, state_below_top_bleu_h, state_below_top_prob_h] + [top_bleu_h, top_prob_h]
                    model.optimizer.loss_value.set_value(1.0)
                for k in range(3):
                    model.fit(x=train_inputs,
                              y=np.zeros((state_below_top_prob_h.shape[0], 1), dtype='float32'),
                              batch_size=min(self.params_training['batch_size'], len(x)),
                              epochs=self.params_training['n_epochs'],
                              verbose=self.params_training['verbose'],
                              callbacks=[],
                              validation_data=None,
                              validation_split=self.params_training.get('val_split', 0.),
                              shuffle=self.params_training['shuffle'],
                              class_weight=None,
                              sample_weight=None,
                              initial_epoch=0)
                    self.n_updates += 1
        else:
            # 2. Post-edit this sample in order to match the reference --> Use y
            # 3. Update net parameters with the corrected samples
            for model in self.models:
                if self.params_training.get('use_custom_loss', False):
                    raise NotImplementedError, 'Custom loss optimizers + train_online (interactive) ' \
                                               'still unimplemented. Refer to sample_and_train_online'

                    weights = model.trainable_weights
                    # weights.sort(key=lambda x: x.name if x.name else x.auto_name)
                    model.optimizer.set_weights(weights)
                    for k in range(1):
                        train_inputs = [x, state_below_y, state_below_h] + [y, hyp]
                        loss_val = model.evaluate(train_inputs,
                                                  np.zeros((y.shape[0], 1), dtype='float32'),
                                                  batch_size=1, verbose=0)
                        loss = 1.0 if loss_val > 0 else 0.0
                        model.optimizer.loss_value.set_value(loss)
                        model.fit(x=train_inputs,
                                  y=np.zeros((y.shape[0], 1), dtype='float32'),
                                  batch_size=min(self.params_training['batch_size'], len(x)),
                                  epochs=self.params_training['n_epochs'],
                                  verbose=self.params_training['verbose'],
                                  callbacks=[],
                                  validation_data=None,
                                  validation_split=self.params_training.get('val_split', 0.),
                                  shuffle=self.params_training['shuffle'],
                                  class_weight=None,
                                  sample_weight=None,
                                  initial_epoch=0)
                        self.n_updates += int(loss)
                else:
                    params = copy.copy(self.params_training)
                    del params['use_custom_loss']
                    del params['n_best_optimizer']
                    del params['optimizer']
                    del params['additional_training_settings']
                    model.trainNetFromSamples([x, state_below_y], y, params)
                    self.n_updates += 1

    def checkParameters(self, input_params, params_training=False):
        """
        Validates a set of input parameters and uses the default ones if not specified.
        :param input_params: Input parameters to validate
        :return:
        """

        default_params_prediction = {'batch_size': 50,
                                     'n_parallel_loaders': 8,
                                     'beam_size': 5,
                                     'normalize': False,
                                     'mean_substraction': True,
                                     'predict_on_sets': ['val'],
                                     'maxlen': 20,
                                     'n_samples': -1,
                                     'model_inputs': ['source_text', 'state_below'],
                                     'model_outputs': ['target_text'],
                                     'dataset_inputs': ['source_text', 'state_below'],
                                     'dataset_outputs': ['target_text'],
                                     'sampling_type': 'max_likelihood',
                                     'words_so_far': False,
                                     'optimized_search': False,
                                     'state_below_index': -1,
                                     'output_text_index': 0,
                                     'store_hypotheses': None,
                                     'search_pruning': False,
                                     'pos_unk': False,
                                     'heuristic': 0,
                                     'mapping': None,
                                     'apply_detokenization': False,
                                     'normalize_probs': False,
                                     'alpha_factor': 1.0,
                                     'coverage_penalty': False,
                                     'length_penalty': False,
                                     'length_norm_factor': 0.0,
                                     'coverage_norm_factor': 0.0,
                                     'output_max_length_depending_on_x': False,
                                     'output_max_length_depending_on_x_factor': 3,
                                     'output_min_length_depending_on_x': False,
                                     'output_min_length_depending_on_x_factor': 2,
                                     'detokenize_f': 'detokenize_none',
                                     'n_best_optimizer': False,
                                     'optimizer_regularizer': 'TER'
                                     }
        default_params_training = {'batch_size': 50,
                                   'use_custom_loss': False,
                                   'optimizer': 'sgd',
                                   'lr': 0.1,
                                   'lr_decay': None,
                                   'lr_gamma': None,
                                   'loss': 'categorical_crossentropy',
                                   'n_best_optimizer': False,
                                   'n_parallel_loaders': 8,
                                   'n_epochs': 1,
                                   'shuffle': False,
                                   'homogeneous_batches': False,
                                   'epochs_for_save': 500,
                                   'verbose': 0,
                                   'eval_on_sets': [],
                                   'extra_callbacks': None,
                                   'reload_epoch': 0,
                                   'epoch_offset': 0,
                                   'data_augmentation': False,
                                   'patience': 0,
                                   'metric_check': None,
                                   'eval_on_epochs': True,
                                   'each_n_epochs': 1,
                                   'start_eval_on_epoch': 0,
                                   'additional_training_settings': {'k': 1, 'tau': 1,
                                                                    'c': 0.5, 'd': 0.5,
                                                                    'lambda': 0.5
                                                                    }
                                   }
        default_params = default_params_training if params_training else default_params_prediction
        valid_params = [key for key in default_params]
        params = dict()

        # Check input parameters' validity
        for key, val in input_params.iteritems():
            if key in valid_params:
                params[key] = val
            else:
                logging.warn("Parameter '" + key + "' is not a valid parameter.")

        # Use default parameters if not provided
        for key, default_val in default_params.iteritems():
            if key not in params:
                params[key] = default_val

        return params

    def get_n_updates(self):
        return self.n_updates
