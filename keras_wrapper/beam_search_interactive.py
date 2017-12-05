# -*- coding: utf-8 -*-

import copy
import logging
import math
import sys
import time

from keras_wrapper.dataset import Data_Batch_Generator
from keras_wrapper.extra.isles_utils import *

logging.basicConfig(level=logging.DEBUG,
                    format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


# logger.setLevel(2)

class BeamSearchEnsemble:
    def __init__(self, models, dataset, params_prediction, n_best=False, verbose=0):
        """

        :param models:
        :param dataset:
        :param params_prediction:
        """
        self.models = models
        self.dataset = dataset
        self.params = params_prediction
        self.optimized_search = params_prediction.get('optimized_search', False)
        self.return_alphas = params_prediction.get('coverage_penalty', False) or params_prediction.get('pos_unk', False)
        self.n_best = n_best
        self.verbose = verbose
        if self.verbose > 0:
            logging.info('<<< "Optimized search: %s >>>' % str(self.optimized_search))

    # PREDICTION FUNCTIONS: Functions for making prediction on input samples
    def predict_cond(self, models, X, states_below, params, ii, prev_outs=None):
        """
        Call the prediction functions of all models, according to their inputs
        :param models: List of models in the ensemble
        :param X: Input data
        :param states_below: Previously generated words (in case of conditional models)
        :param params: Model parameters
        :param ii: Decoding time-step
        :param prev_outs: Only for optimized models. Outputs from the previous time-step.
        :return: Combined outputs from the ensemble
        """

        probs_list = []
        prev_outs_list = []
        alphas_list = []
        for i, model in enumerate(models):
            if self.optimized_search:
                [model_probs, next_outs] = model.predict_cond_optimized(X,
                                                                        states_below,
                                                                        params,
                                                                        ii,
                                                                        prev_out=prev_outs[i])
                probs_list.append(model_probs)
                if self.return_alphas:
                    alphas_list.append(next_outs[-1][0])  # Shape: (k, n_steps)
                    next_outs = next_outs[:-1]
                prev_outs_list.append(next_outs)
            else:
                probs_list.append(model.predict_cond(X, states_below, params, ii))
        probs = sum(probs_list[i] for i in xrange(len(models))) / float(len(models))

        if self.return_alphas:
            alphas = np.asarray(sum(alphas_list[i] for i in xrange(len(models))))
        else:
            alphas = None
        if self.optimized_search:
            return probs, prev_outs_list, alphas
        else:
            return probs

    def beam_search(self, X, params, eos_sym=0, null_sym=2):
        """
        Beam search method for Cond models.
        (https://en.wikibooks.org/wiki/Artificial_Intelligence/Search/Heuristic_search/Beam_search)
        The algorithm in a nutshell does the following:

        1. k = beam_size
        2. open_nodes = [[]] * k
        3. while k > 0:

            3.1. Given the inputs, get (log) probabilities for the outputs.

            3.2. Expand each open node with all possible output.

            3.3. Prune and keep the k best nodes.

            3.4. If a sample has reached the <eos> symbol:

                3.4.1. Mark it as final sample.

                3.4.2. k -= 1

            3.5. Build new inputs (state_below) and go to 1.

        4. return final_samples, final_scores

        :param X: Model inputs
        :param params: Search parameters
        :param null_sym: <null> index
        :param eos_sym: end-of-sentence index
        :return: UNSORTED list of [k_best_samples, k_best_scores] (k: beam size)
        """
        k = params['beam_size']
        samples = []
        sample_scores = []
        pad_on_batch = params['pad_on_batch']
        dead_k = 0  # samples that reached eos
        live_k = 1  # samples that did not yet reached eos
        hyp_samples = [[]] * live_k
        hyp_scores = np.zeros(live_k).astype('float32')
        if self.return_alphas:
            sample_alphas = []
            hyp_alphas = [[]] * live_k

        maxlen = int(len(X[params['dataset_inputs'][0]][0]) * params['output_max_length_depending_on_x_factor']) if \
            params['output_max_length_depending_on_x'] else params['maxlen']

        minlen = int(
            len(X[params['dataset_inputs'][0]][0]) / params['output_min_length_depending_on_x_factor'] + 1e-7) if \
            params['output_min_length_depending_on_x'] else 0

        # we must include an additional dimension if the input for each timestep are all the generated "words_so_far"
        if params['words_so_far']:
            if k > maxlen:
                raise NotImplementedError("BEAM_SIZE can't be higher than MAX_OUTPUT_TEXT_LEN!")
            state_below = np.asarray([[null_sym]] * live_k) \
                if pad_on_batch else np.asarray([np.zeros((maxlen, maxlen))] * live_k)
        else:
            state_below = np.asarray([null_sym] * live_k) \
                if pad_on_batch else np.asarray([np.zeros(maxlen)] * live_k)
        prev_outs = [None] * len(self.models)
        for ii in xrange(maxlen):
            # for every possible live sample calc prob for every possible label
            if self.optimized_search:  # use optimized search model if available
                [probs, prev_outs, alphas] = self.predict_cond(self.models, X, state_below, params, ii,
                                                               prev_outs=prev_outs)
            else:
                probs = self.predict_cond(self.models, X, state_below, params, ii)

            if minlen > 0 and ii < minlen:
                probs[:, eos_sym] = -np.inf

            # total score for every sample is sum of -log of word prb
            cand_scores = np.array(hyp_scores)[:, None] - np.log(probs)
            cand_flat = cand_scores.flatten()
            # Find the best options by calling argsort of flatten array
            ranks_flat = cand_flat.argsort()[:(k - dead_k)]
            # Decypher flatten indices
            voc_size = probs.shape[1]
            trans_indices = ranks_flat / voc_size  # index of row
            word_indices = ranks_flat % voc_size  # index of col
            costs = cand_flat[ranks_flat]
            best_cost = costs[0]
            # Form a beam for the next iteration
            new_hyp_samples = []
            new_trans_indices = []
            new_hyp_scores = np.zeros(k - dead_k).astype('float32')
            if self.return_alphas:
                new_hyp_alphas = []
            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                if params['search_pruning']:
                    if costs[idx] < k * best_cost:
                        new_hyp_samples.append(hyp_samples[ti] + [wi])
                        new_trans_indices.append(ti)
                        new_hyp_scores[idx] = copy.copy(costs[idx])
                        if self.return_alphas:
                            new_hyp_alphas.append(hyp_alphas[ti] + [alphas[ti]])
                    else:
                        dead_k += 1
                else:
                    new_hyp_samples.append(hyp_samples[ti] + [wi])
                    new_trans_indices.append(ti)
                    new_hyp_scores[idx] = copy.copy(costs[idx])
                    if self.return_alphas:
                        new_hyp_alphas.append(hyp_alphas[ti] + [alphas[ti]])

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_alphas = []
            indices_alive = []
            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == eos_sym:  # finished sample
                    samples.append(new_hyp_samples[idx])
                    sample_scores.append(new_hyp_scores[idx])
                    if self.return_alphas:
                        sample_alphas.append(new_hyp_alphas[idx])
                    dead_k += 1
                else:
                    indices_alive.append(new_trans_indices[idx])
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    if self.return_alphas:
                        hyp_alphas.append(new_hyp_alphas[idx])
            hyp_scores = np.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break
            state_below = np.asarray(hyp_samples, dtype='int64')

            # we must include an additional dimension if the input for each timestep are all the generated words so far
            if pad_on_batch:
                state_below = np.hstack((np.zeros((state_below.shape[0], 1), dtype='int64') + null_sym, state_below))
                if params['words_so_far']:
                    state_below = np.expand_dims(state_below, axis=0)
            else:
                state_below = np.hstack((np.zeros((state_below.shape[0], 1), dtype='int64'), state_below,
                                         np.zeros((state_below.shape[0],
                                                   max(maxlen - state_below.shape[1] - 1, 0)),
                                                  dtype='int64')))

                if params['words_so_far']:
                    state_below = np.expand_dims(state_below, axis=0)
                    state_below = np.hstack((state_below,
                                             np.zeros((state_below.shape[0], maxlen - state_below.shape[1],
                                                       state_below.shape[2]))))

            if self.optimized_search and ii > 0:
                for n_model in range(len(self.models)):
                    # filter next search inputs w.r.t. remaining samples
                    for idx_vars in range(len(prev_outs[n_model])):
                        prev_outs[n_model][idx_vars] = prev_outs[n_model][idx_vars][indices_alive]

        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                samples.append(hyp_samples[idx])
                sample_scores.append(hyp_scores[idx])
                if self.return_alphas:
                    sample_alphas.append(hyp_alphas[idx])
        if self.return_alphas:
            return samples, sample_scores, sample_alphas
        else:
            return samples, sample_scores, None

    def predictBeamSearchNetInteractive(self):
        """
        Approximates by beam search the best predictions of the net on the dataset splits chosen.
        Params from config that affect the sarch process:
            * max_batch_size: size of the maximum batch loaded into memory
            * n_parallel_loaders: number of parallel data batch loaders
            * normalization: apply data normalization on images/features or not (only if using images/features as input)
            * mean_substraction: apply mean data normalization on images or not (only if using images as input)
            * predict_on_sets: list of set splits for which we want to extract the predictions ['train', 'val', 'test']
            * optimized_search: boolean indicating if the used model has the optimized Beam Search implemented
             (separate self.model_init and self.model_next models for reusing the information from previous timesteps).

        The following attributes must be inserted to the model when building an optimized search model:

            * ids_inputs_init: list of input variables to model_init (must match inputs to conventional model)
            * ids_outputs_init: list of output variables of model_init (model probs must be the first output)
            * ids_inputs_next: list of input variables to model_next (previous word must be the first input)
            * ids_outputs_next: list of output variables of model_next (model probs must be the first output and
                                the number of out variables must match the number of in variables)
            * matchings_init_to_next: dictionary from 'ids_outputs_init' to 'ids_inputs_next'
            * matchings_next_to_next: dictionary from 'ids_outputs_next' to 'ids_inputs_next'

        :returns predictions: dictionary with set splits as keys and matrices of predictions as values.
        """

        # Check input parameters and recover default values if needed
        default_params = {'max_batch_size': 50,
                          'n_parallel_loaders': 8,
                          'beam_size': 5,
                          'normalize': False,
                          'mean_substraction': True,
                          'predict_on_sets': ['val'],
                          'maxlen': 20,
                          'n_samples': -1,
                          'model_inputs': ['source_text', 'state_below'],
                          'model_outputs': ['description'],
                          'dataset_inputs': ['source_text', 'state_below'],
                          'dataset_outputs': ['description'],
                          'sampling_type': 'max_likelihood',
                          'words_so_far': False,
                          'optimized_search': False,
                          'pos_unk': False,
                          'state_below_index': -1,
                          'search_pruning': False,
                          'normalize_probs': False,
                          'alpha_factor': 0.0,
                          'coverage_penalty': False,
                          'length_penalty': False,
                          'length_norm_factor': 0.0,
                          'coverage_norm_factor': 0.0,
                          'output_max_length_depending_on_x': False,
                          'output_max_length_depending_on_x_factor': 3,
                          'output_min_length_depending_on_x': False,
                          'output_min_length_depending_on_x_factor': 2
                          }
        params = self.checkParameters(self.params, default_params)
        predictions = dict()
        for s in params['predict_on_sets']:
            logging.info("\n <<< Predicting outputs of " + s + " set >>>")
            assert len(params['model_inputs']) > 0, 'We need at least one input!'
            if not params['optimized_search']:  # use optimized search model if available
                assert not self.return_alphas, 'Return alphas is not supported with non-optimized beam search methods'
            params['pad_on_batch'] = self.dataset.pad_on_batch[params['dataset_inputs'][-1]]
            # Calculate how many interations are we going to perform
            if params['n_samples'] < 1:
                n_samples = eval("self.dataset.len_" + s)
                num_iterations = int(math.ceil(float(n_samples)))  # / params['batch_size']))

                # Prepare data generator: We won't use an Homogeneous_Data_Batch_Generator here
                # TODO: We prepare data as model 0... Different data preparators for each model?
                data_gen = Data_Batch_Generator(s,
                                                self.models[0],
                                                self.dataset,
                                                num_iterations,
                                                batch_size=1,
                                                normalization=params['normalize'],
                                                data_augmentation=False,
                                                mean_substraction=params['mean_substraction'],
                                                predict=True)#.generator()
            else:
                n_samples = params['n_samples']
                num_iterations = int(math.ceil(float(n_samples)))  # / params['batch_size']))

                # Prepare data generator: We won't use an Homogeneous_Data_Batch_Generator here
                data_gen = Data_Batch_Generator(s,
                                                self.models[0],
                                                self.dataset,
                                                num_iterations,
                                                batch_size=1,
                                                normalization=params['normalize'],
                                                data_augmentation=False,
                                                mean_substraction=params['mean_substraction'],
                                                predict=False,
                                                random_samples=n_samples)#.generator()
            if params['n_samples'] > 0:
                references = []
                sources_sampling = []
            best_samples = []
            if self.return_alphas:
                best_alphas = []
                sources = []

            total_cost = 0
            sampled = 0
            start_time = time.time()
            eta = -1
            if self.n_best:
                n_best_list = []
            for j in range(num_iterations):
                data = data_gen.next()
                X = dict()
                if params['n_samples'] > 0:
                    s_dict = {}
                    for input_id in params['model_inputs']:
                        X[input_id] = data[0][input_id]
                        s_dict[input_id] = X[input_id]
                    sources_sampling.append(s_dict)

                    Y = dict()
                    for output_id in params['model_outputs']:
                        Y[output_id] = data[1][output_id]
                else:
                    s_dict = {}
                    for input_id in params['model_inputs']:
                        X[input_id] = data[input_id]
                        if self.return_alphas:
                            s_dict[input_id] = X[input_id]
                    if self.return_alphas:
                        sources.append(s_dict)

                for i in range(len(X[params['model_inputs'][0]])):
                    sampled += 1
                    sys.stdout.write('\r')
                    sys.stdout.write("Sampling %d/%d  -  ETA: %ds " % (sampled, n_samples, int(eta)))
                    sys.stdout.flush()
                    x = dict()
                    for input_id in params['model_inputs']:
                        x[input_id] = np.asarray([X[input_id][i]])
                    samples, scores, alphas = self.beam_search(x, params, null_sym=self.dataset.extra_words['<null>'])

                    if params['length_penalty'] or params['coverage_penalty']:
                        if params['length_penalty']:
                            length_penalties = [((5 + len(sample)) ** params['length_norm_factor']
                                                 / (5 + 1) ** params['length_norm_factor'])
                                                # this 5 is a magic number by Google...
                                                for sample in samples]
                        else:
                            length_penalties = [1.0 for _ in len(samples)]

                        if params['coverage_penalty']:
                            coverage_penalties = []
                            for k, sample in enumerate(samples):
                                # We assume that source sentences are at the first position of x
                                x_sentence = x[params['model_inputs'][0]][0]
                                alpha = np.asarray(alphas[k])
                                cp_penalty = 0.0
                                for cp_i in range(len(x_sentence)):
                                    att_weight = 0.0
                                    for cp_j in range(len(sample)):
                                        att_weight += alpha[cp_j, cp_i]
                                    cp_penalty += np.log(min(att_weight, 1.0))
                                coverage_penalties.append(params['coverage_norm_factor'] * cp_penalty)
                        else:
                            coverage_penalties = [0.0 for _ in len(samples)]
                        scores = [co / lp + cp for co, lp, cp in zip(scores, length_penalties, coverage_penalties)]

                    elif params['normalize_probs']:
                        counts = [len(sample) ** params['alpha_factor'] for sample in samples]
                        scores = [co / cn for co, cn in zip(scores, counts)]

                    if self.n_best:
                        n_best_indices = np.argsort(scores)
                        n_best_scores = np.asarray(scores)[n_best_indices]
                        n_best_samples = np.asarray(samples)[n_best_indices]
                        if alphas is not None:
                            n_best_alphas = [np.stack(alphas[i]) for i in n_best_indices]
                        else:
                            n_best_alphas = [None] * len(n_best_indices)
                        n_best_list.append([n_best_samples, n_best_scores, n_best_alphas])
                    best_score = np.argmin(scores)
                    best_sample = samples[best_score]
                    best_samples.append(best_sample)
                    if self.return_alphas:
                        best_alphas.append(np.asarray(alphas[best_score]))
                    total_cost += scores[best_score]
                    eta = (n_samples - sampled) * (time.time() - start_time) / sampled
                    if params['n_samples'] > 0:
                        for output_id in params['model_outputs']:
                            references.append(Y[output_id][i])

            sys.stdout.write('Total cost of the translations: %f \t '
                             'Average cost of the translations: %f\n' % (total_cost, total_cost / n_samples))
            sys.stdout.write('The sampling took: %f secs (Speed: %f sec/sample)\n' %
                             ((time.time() - start_time), (time.time() - start_time) / n_samples))

            sys.stdout.flush()
            if self.n_best:
                if self.return_alphas:
                    predictions[s] = (np.asarray(best_samples), np.asarray(best_alphas), sources), n_best_list
                else:
                    predictions[s] = np.asarray(best_samples), n_best_list
            else:
                if self.return_alphas:
                    predictions[s] = (np.asarray(best_samples), np.asarray(best_alphas), sources)
                else:
                    predictions[s] = np.asarray(best_samples)

        if params['n_samples'] < 1:
            return predictions
        else:
            return predictions, references, sources_sampling

    def sample_beam_search(self, src_sentence):
        """

        :param src_sentence:
        :return:
        """
        # Check input parameters and recover default values if needed
        default_params = {'max_batch_size': 50,
                          'n_parallel_loaders': 8,
                          'beam_size': 5,
                          'normalize': False,
                          'mean_substraction': True,
                          'predict_on_sets': ['val'],
                          'maxlen': 20,
                          'n_samples': 1,
                          'model_inputs': ['source_text', 'state_below'],
                          'model_outputs': ['description'],
                          'dataset_inputs': ['source_text', 'state_below'],
                          'dataset_outputs': ['description'],
                          'sampling_type': 'max_likelihood',
                          'words_so_far': False,
                          'optimized_search': False,
                          'state_below_index': -1,
                          'output_text_index': 0,
                          'search_pruning': False,
                          'pos_unk': False,
                          'normalize_probs': False,
                          'alpha_factor': 0.0,
                          'coverage_penalty': False,
                          'length_penalty': False,
                          'length_norm_factor': 0.0,
                          'coverage_norm_factor': 0.0,
                          'output_max_length_depending_on_x': False,
                          'output_max_length_depending_on_x_factor': 3,
                          'output_min_length_depending_on_x': False,
                          'output_min_length_depending_on_x_factor': 2
                          }
        params = self.checkParameters(self.params, default_params)
        params['pad_on_batch'] = self.dataset.pad_on_batch[params['dataset_inputs'][-1]]
        params['n_samples'] = 1
        if self.n_best:
            n_best_list = []
        X = dict()
        for input_id in params['model_inputs']:
            X[input_id] = src_sentence
        x = dict()
        for input_id in params['model_inputs']:
            x[input_id] = np.asarray([X[input_id]])
        samples, scores, alphas = self.beam_search(x,
                                                   params,
                                                   null_sym=self.dataset.extra_words['<null>'])

        if params['length_penalty'] or params['coverage_penalty']:
            if params['length_penalty']:
                length_penalties = [((5 + len(sample)) ** params['length_norm_factor']
                                     / (5 + 1) ** params['length_norm_factor'])  # this 5 is a magic number by Google...
                                    for sample in samples]
            else:
                length_penalties = [1.0 for _ in len(samples)]

            if params['coverage_penalty']:
                coverage_penalties = []
                for k, sample in enumerate(samples):
                    # We assume that source sentences are at the first position of x
                    x_sentence = x[params['model_inputs'][0]][0]
                    alpha = np.asarray(alphas[k])
                    cp_penalty = 0.0
                    for cp_i in range(len(x_sentence)):
                        att_weight = 0.0
                        for cp_j in range(len(sample)):
                            att_weight += alpha[cp_j, cp_i]
                        cp_penalty += np.log(min(att_weight, 1.0))
                    coverage_penalties.append(params['coverage_norm_factor'] * cp_penalty)
            else:
                coverage_penalties = [0.0 for _ in len(samples)]
            scores = [co / lp + cp for co, lp, cp in zip(scores, length_penalties, coverage_penalties)]

        elif params['normalize_probs']:
            counts = [len(sample) ** params['alpha_factor'] for sample in samples]
            scores = [co / cn for co, cn in zip(scores, counts)]

        if self.n_best:
            n_best_indices = np.argsort(scores)
            n_best_scores = np.asarray(scores)[n_best_indices]
            n_best_samples = np.asarray(samples)[n_best_indices]
            if alphas is not None:
                n_best_alphas = [np.stack(alphas[i]) for i in n_best_indices]
            else:
                n_best_alphas = [None] * len(n_best_indices)
            n_best_list.append([n_best_samples, n_best_scores, n_best_alphas])

        best_score_idx = np.argmin(scores)
        best_sample = samples[best_score_idx]
        if self.return_alphas:
            best_alphas = np.asarray(alphas[best_score_idx])
        else:
            best_alphas = None
        if self.n_best:
            return (np.asarray(best_sample), scores[best_score_idx], np.asarray(best_alphas)), n_best_list
        else:
            return np.asarray(best_sample), scores[best_score_idx], np.asarray(best_alphas)

    @staticmethod
    def checkParameters(input_params, default_params):
        """
            Validates a set of input parameters and uses the default ones if not specified.
        """
        valid_params = [key for key in default_params]
        params = dict()

        # Check input parameters' validity
        for key, val in input_params.iteritems():
            if key in valid_params:
                params[key] = val
            else:
                logging.warn("Unexpected parameter: '" + key + "'.")

        # Use default parameters if not provided
        for key, default_val in default_params.iteritems():
            if key not in params:
                params[key] = default_val

        return params


class InteractiveBeamSearchSampler:
    def __init__(self, models, dataset, params_prediction, excluded_words=None, verbose=0):
        """
        Class for sampling taking into account the user's feedback
        :param models:
        :param dataset:
        :param params_prediction:
        :param excluded_words: Indices excluded for the search (in the segment-based approach).
                               A typical usage: Exclude non-terminal subwords if we are in the last position of the
                               middle of two validated segments.
        :param verbose:
        """
        self.models = models
        self.dataset = dataset
        self.params = params_prediction
        self.optimized_search = params_prediction.get('optimized_search', False)
        self.return_alphas = params_prediction.get('coverage_penalty', False) or params_prediction.get('pos_unk', False)
        self.verbose = verbose
        self.n_best = False  # TODO: Useless attribute (for the moment...)
        self.excluded_words = np.asarray(excluded_words, dtype='int8') if excluded_words is not None else excluded_words
        if self.verbose > 0:
            logging.info('<<< "Optimized search: %s >>>' % str(self.optimized_search))

    # PREDICTION FUNCTIONS: Functions for making prediction on input samples
    def predict_cond(self, models, X, states_below, params, ii, prev_outs=None):
        """
        Call the prediction functions of all models, according to their inputs
        :param models: List of models in the ensemble
        :param X: Input data
        :param states_below: Previously generated words (in case of conditional models)
        :param params: Model parameters
        :param ii: Decoding time-step
        :param prev_outs: Only for optimized models. Outputs from the previous time-step.
        :return: Combined outputs from the ensemble
        """

        probs_list = []
        prev_outs_list = []
        alphas_list = []
        for i, model in enumerate(models):
            if self.optimized_search:
                [model_probs, next_outs] = model.predict_cond_optimized(X,
                                                                        states_below,
                                                                        params,
                                                                        ii,
                                                                        prev_out=prev_outs[i])
                probs_list.append(model_probs)
                if self.return_alphas:
                    alphas_list.append(next_outs[-1][0])  # Shape: (k, n_steps)
                    next_outs = next_outs[:-1]
                prev_outs_list.append(next_outs)
            else:
                probs_list.append(model.predict_cond(X, states_below, params, ii))
        probs = sum(probs_list[i] for i in xrange(len(models))) / float(len(models))

        if self.return_alphas:
            alphas = np.asarray(sum(alphas_list[i] for i in xrange(len(models))))
        else:
            alphas = None
        if self.optimized_search:
            return probs, prev_outs_list, alphas
        else:
            return probs

    def interactive_beam_search(self, X, params, fixed_words=None, max_N=0, isles=None,
                                valid_next_words=None, eos_sym=0, null_sym=2, idx2word=None):
        """
        Beam search method for Cond models.
        (https://en.wikibooks.org/wiki/Artificial_Intelligence/Search/Heuristic_search/Beam_search)
        The algorithm in a nutshell does the following:

        1. k = beam_size
        2. open_nodes = [[]] * k
        3. while k > 0:

            3.1. Given the inputs, get (log) probabilities for the outputs.

            3.2. Expand each open node with all possible output.

            3.3. Prune and keep the k best nodes.

            3.4. If a sample has reached the <eos> symbol:

                3.4.1. Mark it as final sample.

                3.4.2. k -= 1

            3.5. Build new inputs (state_below) and go to 1.

        4. return final_samples, final_scores
        :param X: Model inputs
        :param params: Search parameters
        :param fixed_words: Dictionary of words fixed by the user: {position: word}
        :param max_N: Maximum number of words to generate between two isles.
        :param isles: Isles fixed by the user. List of (isle_index, [words]) (Although isle_index is never used
        :param valid_next_words: List of candidate words to be the next one to generate (after generating fixed_words)
        :param eos_sym: End-of-sentence index
        :param idx2word:  Mapping between indices and words
        :param null_sym: <null> symbol
        :return: UNSORTED list of [k_best_samples, k_best_scores] (k: beam size)
        """

        if fixed_words is None:
            fixed_words = dict()
        if isles is None:
            isles = list()
        if idx2word is None:
            idx2word = dict()

        k = params['beam_size']
        ii = 0
        samples = []
        sample_scores = []
        pad_on_batch = params['pad_on_batch']
        dead_k = 0  # samples that reached eos
        live_k = 1  # samples that did not yet reached eos
        hyp_samples = [[]] * live_k
        hyp_scores = np.zeros(live_k).astype('float32')

        if isles is not None:
            # unfixed_isles = filter(lambda x: not is_sublist(x[1], fixed_words.values()),
            # [segment for segment in isles])
            fixed_words_v = copy.copy(fixed_words.values())
            unfixed_isles = []
            for segment in isles:
                if is_sublist(segment[1], fixed_words_v):
                    s, starting_pos = subfinder(segment[1], fixed_words_v)
                    for i in range(len(s)):
                        del fixed_words_v[starting_pos]
                else:
                    unfixed_isles.append(segment)
            logger.log(3, 'Unfixed isles: ' + str(unfixed_isles))
        else:
            unfixed_isles = []

        if self.return_alphas:
            sample_alphas = []
            hyp_alphas = [[]] * live_k

        len_fixed_words = len(fixed_words.keys())

        maxlen = int(len(X[params['dataset_inputs'][0]][0]) * params['output_max_length_depending_on_x_factor']) if \
            params['output_max_length_depending_on_x'] else params['maxlen']
        if maxlen < len_fixed_words:
            maxlen += len_fixed_words

        minlen = int(
            len(X[params['dataset_inputs'][0]][0]) / params['output_min_length_depending_on_x_factor'] + 1e-7) if \
            params['output_min_length_depending_on_x'] else 0

        state_below = np.asarray([null_sym] * live_k) \
            if pad_on_batch else np.asarray([np.zeros(params['maxlen'])] * live_k)

        prev_outs = [None] * len(self.models)
        if valid_next_words is not None:
            # We need to generate at least the partial hypothesis provided by the user
            minlen += 1
            maxlen += 1

        while ii <= maxlen:
            # for every possible live sample calc prob for every possible label
            logger.log(2, "hyp_samples" + str(hyp_samples))
            logger.log(2, "hyp_scores" + str(hyp_scores))
            logger.log(3, "ii: %d, fixed_words= %s" % (ii, str(fixed_words)))
            logger.log(3, 'Current beam:' + str(['<<< Hypo ' + str(i) + ': ' +
                                                 str(map(lambda word_: idx2word.get(word_, "UNK"), hyp)) +
                                                 ' >>>' for (i, hyp) in enumerate(hyp_samples)]))
            if self.optimized_search:  # use optimized search model if available
                [probs, prev_outs, alphas] = self.predict_cond(self.models,
                                                               X,
                                                               state_below,
                                                               params,
                                                               ii,
                                                               prev_outs=prev_outs)
            else:
                probs = self.predict_cond(self.models,
                                          X,
                                          state_below,
                                          params,
                                          ii)
            # total score for every sample is sum of -log of word prb
            log_probs = np.log(probs)
            # Adjust log probs according to search restrictions
            if len_fixed_words == 0:
                max_fixed_pos = 0
            else:
                max_fixed_pos = max(fixed_words.keys())

            if minlen > 0 and ii < minlen:
                probs[:, eos_sym] = -np.inf

            if len(unfixed_isles) > 0 or ii <= max_fixed_pos:
                log_probs[:, eos_sym] = -np.inf

            if valid_next_words is not None and ii == len_fixed_words:
                logger.log(3, 'valid_next_words: ' + str(valid_next_words))
                if valid_next_words != dict():
                    next_word_antiprefix = [idx for idx in idx2word.keys() if idx not in valid_next_words]
                    log_probs[:, next_word_antiprefix] = -np.inf
                log_probs[:, eos_sym] = -np.inf

            if len(unfixed_isles) == 0 or ii in fixed_words:  # There are no remaining isles. Regular decoding.
                # If word is fixed, we only consider this hypothesis
                if ii in fixed_words:
                    trans_indices = range(len(hyp_samples))
                    word_indices = [fixed_words[ii]] * len(trans_indices)
                    costs = np.array(hyp_scores)

                else:
                    # Decypher flatten indices
                    cand_scores = np.array(hyp_scores)[:, None] - log_probs
                    cand_flat = cand_scores.flatten()
                    # Find the best options by calling argsort of flatten array
                    ranks_flat = cand_flat.argsort()[:(k - dead_k)]
                    voc_size = probs.shape[1]
                    trans_indices = ranks_flat / voc_size  # index of row
                    word_indices = ranks_flat % voc_size  # index of col
                    costs = cand_flat[ranks_flat]

                # Form a beam for the next iteration
                new_hyp_samples = []
                new_trans_indices = []
                new_hyp_scores = np.zeros(k - dead_k).astype('float32')
                if self.return_alphas:
                    new_hyp_alphas = []

                for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                    if params['search_pruning']:
                        if costs[idx] < k * best_cost:
                            new_hyp_samples.append(hyp_samples[ti] + [wi])
                            new_trans_indices.append(ti)
                            new_hyp_scores[idx] = copy.copy(costs[idx])
                            if self.return_alphas:
                                new_hyp_alphas.append(hyp_alphas[ti] + [alphas[ti]])
                        else:
                            dead_k += 1
                    else:
                        new_hyp_samples.append(hyp_samples[ti] + [wi])
                        new_trans_indices.append(ti)
                        new_hyp_scores[idx] = copy.copy(costs[idx])
                        if self.return_alphas:
                            new_hyp_alphas.append(hyp_alphas[ti] + [alphas[ti]])

                # check the finished samples
                new_live_k = 0
                hyp_samples = []
                hyp_scores = []
                hyp_alphas = []
                indices_alive = []
                for idx in xrange(len(new_hyp_samples)):
                    if new_hyp_samples[idx][-1] == eos_sym:  # finished sample
                        samples.append(new_hyp_samples[idx])
                        sample_scores.append(new_hyp_scores[idx])
                        if self.return_alphas:
                            sample_alphas.append(new_hyp_alphas[idx])
                        dead_k += 1
                    else:
                        indices_alive.append(new_trans_indices[idx])
                        new_live_k += 1
                        hyp_samples.append(new_hyp_samples[idx])
                        hyp_scores.append(new_hyp_scores[idx])
                        if self.return_alphas:
                            hyp_alphas.append(new_hyp_alphas[idx])
                hyp_scores = np.array(hyp_scores)
                live_k = new_live_k

                if new_live_k < 1:
                    break
                if dead_k >= k:
                    break
                state_below = np.asarray(hyp_samples, dtype='int64')

                if pad_on_batch:
                    state_below = np.hstack((np.zeros((state_below.shape[0], 1), dtype='int64') + null_sym,
                                             state_below))
                else:
                    state_below = np.hstack((np.zeros((state_below.shape[0], 1), dtype='int64'),
                                             state_below,
                                             np.zeros((state_below.shape[0],
                                                       max(maxlen - state_below.shape[1] - 1, 0)),
                                                      dtype='int64')))

                if self.optimized_search and ii > 0:
                    for n_model in range(len(self.models)):
                        # filter next search inputs w.r.t. remaining samples
                        for idx_vars in range(len(prev_outs[n_model])):
                            prev_outs[n_model][idx_vars] = prev_outs[n_model][idx_vars][indices_alive]
            else:  # We are in the middle of two isles
                forward_hyp_trans = [[]] * max_N
                forward_hyp_scores = [[]] * max_N
                if self.return_alphas:
                    forward_alphas = [[]] * max_N
                forward_state_belows = [[]] * max_N
                forward_prev_outs = [[]] * max_N
                forward_indices_alive = [[]] * max_N

                hyp_samples_ = copy.copy(hyp_samples)
                hyp_scores_ = copy.copy(hyp_scores)
                if self.return_alphas:
                    hyp_alphas_ = copy.copy(hyp_alphas)
                n_samples_ = k - dead_k
                for forward_steps in range(max_N):
                    if self.optimized_search:  # use optimized search model if available
                        [probs, prev_outs, alphas] = self.predict_cond(self.models,
                                                                       X,
                                                                       state_below,
                                                                       params,
                                                                       ii + forward_steps,
                                                                       prev_outs=prev_outs)
                    else:
                        probs = self.predict_cond(self.models,
                                                  X,
                                                  state_below,
                                                  params,
                                                  ii + forward_steps)
                    # total score for every sample is sum of -log of word prb
                    log_probs = np.log(probs)

                    # Adjust log probs according to search restrictions
                    log_probs[:, eos_sym] = -np.inf

                    # if excluded words:
                    if self.excluded_words is not None:
                        allowed_log_probs = copy.copy(log_probs)
                        allowed_log_probs[:, self.excluded_words] = -np.inf

                    # If word is fixed, we only consider this hypothesis
                    if ii + forward_steps in fixed_words:
                        trans_indices = range(n_samples_)
                        word_indices = [fixed_words[ii + forward_steps]] * len(trans_indices)
                        costs = np.array(hyp_scores_)
                    else:
                        # Decypher flatten indices
                        next_costs = np.array(hyp_scores_)[:, None] - log_probs
                        flat_next_costs = next_costs.flatten()
                        # Find the best options by calling argsort of flatten array
                        ranks_flat = flat_next_costs.argsort()[:n_samples_]
                        voc_size = probs.shape[1]
                        trans_indices = ranks_flat / voc_size  # index of row
                        word_indices = ranks_flat % voc_size  # index of col
                        costs = flat_next_costs[ranks_flat]
                        if self.excluded_words is not None:
                            allowed_next_costs = np.array(hyp_scores_)[:, None] - allowed_log_probs
                            allowed_flat_next_costs = allowed_next_costs.flatten()
                            allowed_ranks_flat = allowed_flat_next_costs.argsort()[:n_samples_]
                            allowed_trans_indices = allowed_ranks_flat / voc_size  # index of row
                            allowed_word_indices = allowed_ranks_flat % voc_size  # index of col
                            allowed_costs = allowed_flat_next_costs[allowed_ranks_flat]

                    # Form a beam for the next iteration
                    new_hyp_samples = []
                    new_trans_indices = []
                    new_hyp_scores = np.zeros(n_samples_).astype('float32')
                    if self.return_alphas:
                        new_hyp_alphas = []
                    for idx, [orig_idx, next_word] in enumerate(zip(trans_indices, word_indices)):
                        new_hyp_samples.append(hyp_samples_[orig_idx] + [next_word])
                        new_trans_indices.append(orig_idx)
                        new_hyp_scores[idx] = copy.copy(costs[idx])
                        if self.return_alphas:
                            new_hyp_alphas.append(hyp_alphas_[orig_idx] + [alphas[orig_idx]])

                    # check the finished samples
                    new_live_k_ = 0
                    hyp_samples_ = []
                    hyp_scores_ = []
                    hyp_alphas_ = []
                    indices_alive_ = []
                    for idx in xrange(len(new_hyp_samples)):
                        indices_alive_.append(new_trans_indices[idx])
                        new_live_k_ += 1
                        hyp_samples_.append(new_hyp_samples[idx])
                        hyp_scores_.append(new_hyp_scores[idx])
                        if self.return_alphas:
                            hyp_alphas_.append(new_hyp_alphas[idx])



                    # Form a beam of allowed hypos for the final evaluation
                    if self.excluded_words is not None:
                        allowed_new_hyp_samples = []
                        allowed_new_trans_indices = []
                        allowed_new_hyp_scores = np.zeros(n_samples_).astype('float32')
                        if self.return_alphas:
                            allowed_new_hyp_alphas = []
                        for idx, [orig_idx, next_word] in enumerate(zip(allowed_trans_indices, allowed_word_indices)):
                            allowed_new_hyp_samples.append(hyp_samples_[orig_idx] + [next_word])
                            allowed_new_trans_indices.append(orig_idx)
                            allowed_new_hyp_scores[idx] = copy.copy(allowed_costs[idx])
                            if self.return_alphas:
                                allowed_new_hyp_alphas.append(hyp_alphas_[orig_idx] + [alphas[orig_idx]])

                        # check the finished samples
                        allowed_hyp_samples_ = []
                        allowed_hyp_scores_ = []
                        allowed_hyp_alphas_ = []
                        allowed_indices_alive_ = []
                        for idx in xrange(len(allowed_new_hyp_samples)):
                            allowed_indices_alive_.append(allowed_new_trans_indices[idx])
                            allowed_hyp_samples_.append(allowed_new_hyp_samples[idx])
                            allowed_hyp_scores_.append(allowed_new_hyp_scores[idx])
                            if self.return_alphas:
                                allowed_hyp_alphas_.append(allowed_new_hyp_alphas[idx])
                    else:
                        allowed_indices_alive_ = indices_alive_
                        allowed_hyp_scores_ = hyp_scores_
                        allowed_hyp_samples_ = hyp_samples_


                    state_below = np.asarray(hyp_samples_, dtype='int64')
                    if pad_on_batch:
                        state_below = np.hstack((np.zeros((state_below.shape[0], 1), dtype='int64') + null_sym,
                                                 state_below))
                    else:
                        state_below = np.hstack((np.zeros((state_below.shape[0], 1), dtype='int64'), state_below,
                                                 np.zeros((state_below.shape[0],
                                                           max(maxlen - state_below.shape[1] - 1, 0)),
                                                          dtype='int64')))
                    forward_indices_alive[forward_steps] = allowed_indices_alive_ #indices_alive_
                    forward_hyp_scores[forward_steps] = allowed_hyp_scores_#hyp_scores_
                    forward_hyp_trans[forward_steps] = allowed_hyp_samples_ #hyp_samples_
                    if self.return_alphas:
                        forward_alphas[forward_steps] = hyp_alphas_
                    forward_state_belows[forward_steps] = state_below
                    forward_prev_outs[forward_steps] = prev_outs
                    if self.optimized_search and ii > 0:
                        for n_model in range(len(self.models)):
                            # filter next search inputs w.r.t. remaining samples
                            for idx_vars in range(len(prev_outs[n_model])):
                                prev_outs[n_model][idx_vars] = prev_outs[n_model][idx_vars][indices_alive_]
                    else:
                        prev_outs = [None] * len(self.models)

                # We get the beam which contains the best hypothesis
                best_n_words = -1
                min_cost = np.inf
                best_hyp = []
                for n_words in range(len(forward_hyp_scores)):
                    for beam_index in range(len(forward_hyp_scores[n_words])):
                        normalized_cost = forward_hyp_scores[n_words][beam_index] / (n_words + 1)
                        if normalized_cost < min_cost:
                            min_cost = normalized_cost
                            best_n_words = n_words
                            best_hyp = forward_hyp_trans[n_words][beam_index]
                            # best_beam_index = beam_index

                assert best_n_words > -1, "Error in the rescoring approach"
                prev_hyp = map(lambda x: idx2word.get(x, "UNK"), hyp_samples[0])

                # We fix the words of the next segment
                stop = False
                best_n_words_index = best_n_words
                logger.log(3, "Generating %d words from position %d" % (best_n_words, ii))
                best_n_words += 1

                while not stop and len(unfixed_isles) > 0:
                    overlapping_position = -1
                    ii_counter = ii + best_n_words
                    next_isle = unfixed_isles[0][1]
                    isle_prefixes = [next_isle[:i + 1] for i in range(len(next_isle))]
                    hyp = map(lambda x: idx2word.get(x, "UNK"), best_hyp)
                    _, start_pos = subfinder(next_isle, best_hyp[-len(next_isle):])
                    if start_pos > -1:  # If the segment is not completely included in the partial hypothesis
                        start_pos += len(best_hyp) - len(next_isle)  # We get its absolute index value

                    logger.log(2, "Previous best hypothesis:%s" % str([(prev_hyp[i], i) for i in range(len(prev_hyp))]))
                    logger.log(4, "Best hypothesis in beam:%s" % str([(hyp[i], i) for i in range(len(hyp))]))
                    logger.log(4, "Current segment: %s\n" % str(map(lambda x: idx2word.get(x, "UNK"), next_isle)))
                    logger.log(4, "Checked against: %s\n" % str(
                        map(lambda x: idx2word.get(x, "UNK"), best_hyp[-len(next_isle):])))
                    logger.log(4, "Start_pos: %s\n" % str(start_pos))

                    case = 0
                    # Detect the case of the following segment
                    if start_pos > -1:  # Segment completely included in the partial hypothesis
                        ii_counter = start_pos
                        case = 1
                        logger.log(4, "Detected case 1: Segment included in hypothesis (position %d)" % ii_counter)
                    else:
                        for i in range(len(best_hyp)):
                            if any(map(lambda x: x == best_hyp[i:], isle_prefixes)):
                                # Segment overlaps with the hypothesis: Remove segment
                                ii_counter = i
                                overlapping_position = i
                                stop = True
                                case = 2
                                logger.log(4, "Detected case 2: Segment overlapped (position %d)" % ii_counter)
                                break

                    if ii_counter == ii + best_n_words:
                        #  Segment not included nor overlapped. We should put the segment after the partial hypothesis
                        logger.log(4, "Detected case 0: Segment not included nor overlapped")
                        case = 0
                        stop = True
                        # ii_counter -= 1

                    new_live_k = 0
                    hyp_samples = []
                    hyp_scores = []
                    if self.return_alphas:
                        hyp_alphas = []
                    state_below = []
                    prev_outs = [[]] * len(self.models)
                    forward_indices_compatible = []

                    # Form a beam with those hypotheses compatible with best_hyp
                    if case == 0:
                        # The segment is not included in the predicted sequence.
                        # Fix the segment next to the current beam
                        logger.log(3, "Treating case 0. The segment is not included in the predicted sequence. "
                                      "Fix the segment next to the current beam")
                        beam_index = 0
                        for beam_index in range(len(forward_hyp_trans[best_n_words_index])):
                            incompatible = False
                            logger.log(2, "Beam_index:" + str(beam_index))
                            for future_ii in range(len(forward_hyp_trans[best_n_words_index][beam_index])):
                                logger.log(2, "Checking index " + str(future_ii))
                                logger.log(2,
                                           "From hypothesis " + str(forward_hyp_trans[best_n_words_index][beam_index]))
                                if fixed_words.get(future_ii) is not None and fixed_words[future_ii] != \
                                        forward_hyp_trans[best_n_words_index][beam_index][future_ii]:
                                    incompatible = True
                                    logger.log(2, "Incompatible!")

                            if not incompatible:
                                forward_indices_compatible.append(beam_index)
                                hyp_samples.append(forward_hyp_trans[best_n_words_index][beam_index])
                                hyp_scores.append(forward_hyp_scores[best_n_words_index][beam_index])
                                new_live_k += 1
                                if self.return_alphas:
                                    hyp_alphas.append(forward_alphas[best_n_words_index][beam_index])
                                state_below.append(forward_state_belows[best_n_words_index][beam_index])
                        logger.log(3, "forward_indices_compatible" + str(forward_indices_compatible))
                        for n_model in range(len(self.models)):
                            prev_outs[n_model] = [[]] * len(forward_prev_outs[best_n_words_index][n_model])
                            # filter next search inputs w.r.t. remaining samples
                            for idx_vars in range(len(forward_prev_outs[best_n_words_index][n_model])):
                                prev_outs[n_model][idx_vars] = forward_prev_outs[best_n_words_index][n_model][idx_vars][
                                    forward_indices_compatible]
                        if len(forward_indices_compatible) == 0:
                            hyp_samples = forward_hyp_trans[best_n_words_index]
                            hyp_scores = forward_hyp_scores[best_n_words_index]
                            if self.return_alphas:
                                hyp_alphas = forward_alphas[best_n_words_index]
                            state_below = forward_state_belows[best_n_words_index]
                            prev_outs = forward_prev_outs[best_n_words_index]
                        state_below = np.array(state_below)
                    if case == 1:
                        #  The segment is included in the hypothesis
                        logger.log(3, "Treating case 1: The segment is included in the hypothesis")
                        logger.log(3, "best_n_words:" + str(best_n_words))
                        logger.log(3, "len(forward_hyp_trans):" + str(len(forward_hyp_trans)))
                        for beam_index in range(len(forward_hyp_trans[best_n_words_index])):
                            _, start_pos = subfinder(next_isle, forward_hyp_trans[best_n_words_index][beam_index])
                            if start_pos > -1:
                                # Compatible with best hypothesis
                                logger.log(3, "Best Hypo: ")
                                logger.log(3, "%s" % str([(hyp[i], i) for i in range(
                                    len(forward_hyp_trans[best_n_words_index][best_n_words_index]))]))
                                logger.log(3, "compatible with")
                                logger.log(3, "%s" % str([(hyp[i], i) for i in range(
                                    len(forward_hyp_trans[best_n_words_index][beam_index]))]))
                                logger.log(3, "(Adding %s words)" % str(best_n_words_index))
                                hyp_samples.append(forward_hyp_trans[best_n_words_index][beam_index])
                                hyp_scores.append(forward_hyp_scores[best_n_words_index][beam_index])
                                forward_indices_compatible.append(forward_indices_alive[best_n_words_index][beam_index])
                                new_live_k += 1
                                if self.return_alphas:
                                    hyp_alphas.append(forward_alphas[best_n_words_index][beam_index])
                                state_below.append(forward_state_belows[best_n_words_index][beam_index])
                        state_below = np.array(state_below)
                        logger.log(2, "forward_indices_compatible" + str(forward_indices_compatible))
                        for n_model in range(len(self.models)):
                            prev_outs[n_model] = [[]] * len(forward_prev_outs[best_n_words_index][n_model])
                            # filter next search inputs w.r.t. remaining samples
                            for idx_vars in range(len(forward_prev_outs[best_n_words_index][n_model])):
                                prev_outs[n_model][idx_vars] = forward_prev_outs[best_n_words_index][n_model][idx_vars][
                                    forward_indices_compatible]
                    if case == 2:
                        #  The segment is overlapped with the hypothesis
                        logger.log(3, "Treating case 2: The segment is overlapped with the hypothesis")
                        assert overlapping_position > -1, 'Error detecting overlapped position!'
                        for beam_index in range(len(forward_hyp_trans[best_n_words_index])):
                            if any(map(lambda x: x == forward_hyp_trans[best_n_words_index][beam_index][
                                                      overlapping_position:],
                                       isle_prefixes)):
                                # Compatible with best hypothesis
                                hyp_samples.append(forward_hyp_trans[best_n_words_index][beam_index])
                                hyp_scores.append(forward_hyp_scores[best_n_words_index][beam_index])
                                forward_indices_compatible.append(forward_indices_alive[best_n_words_index][beam_index])
                                new_live_k += 1
                                logger.log(3, "Best Hypo: ")
                                logger.log(3, "%s" % str([(hyp[i], i) for i in range(
                                    len(forward_hyp_trans[best_n_words_index][best_n_words_index]))]))
                                logger.log(3, "compatible with")
                                logger.log(3, "%s" % str([(hyp[i], i) for i in range(
                                    len(forward_hyp_trans[best_n_words_index][beam_index]))]))
                                logger.log(3, "(Adding %s words)" % str(best_n_words_index))
                                if self.return_alphas:
                                    hyp_alphas.append(forward_alphas[best_n_words_index][beam_index])
                                state_below.append(forward_state_belows[best_n_words_index][beam_index])
                        state_below = np.array(state_below)
                        logger.log(2, "forward_indices_compatible" + str(forward_indices_compatible))
                        for n_model in range(len(self.models)):
                            prev_outs[n_model] = [[]] * len(forward_prev_outs[best_n_words_index][n_model])
                            # filter next search inputs w.r.t. remaining samples
                            for idx_vars in range(len(forward_prev_outs[best_n_words_index][n_model])):
                                prev_outs[n_model][idx_vars] = forward_prev_outs[best_n_words_index][n_model][idx_vars][
                                    forward_indices_compatible]

                    for word in next_isle:
                        if fixed_words.get(ii_counter) is None:
                            fixed_words[ii_counter] = word
                            logger.log(4, "\t > Word %s (%d) will go to position %d" % (
                                idx2word.get(word, "UNK"), word, ii_counter))
                        else:
                            logger.log(4, "\t > Can't put word %s (%d) in position %d because it is in fixed_words" % (
                                idx2word.get(word, "UNK"), word, ii_counter))
                        ii_counter += 1
                    del unfixed_isles[0]
                    logger.log(3, "\t Stop: >" + str(stop))

                logger.log(4, 'After looking into the future:' +
                           str(['<<< Hypo ' + str(i) + ': ' + str(map(lambda word_: idx2word.get(word_, "UNK"), hyp)) +
                                ' >>>' for (i, hyp) in enumerate(hyp_samples)]))
                ii += best_n_words - 1
                live_k = new_live_k
            ii += 1
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                samples.append(hyp_samples[idx])
                sample_scores.append(hyp_scores[idx])
                if self.return_alphas:
                    sample_alphas.append(hyp_alphas[idx])
        if self.return_alphas:
            return samples, sample_scores, sample_alphas
        else:
            return samples, sample_scores, None

    def sample_beam_search_interactive(self, src_sentence, fixed_words=None, max_N=0,
                                       isles=None, valid_next_words=None, idx2word=None):
        """
        Samples a sentence using the restrictions provided in fixed_words.
        :param src_sentence: Source sentence to translate.
        :param fixed_words: Dictionary of words fixed by the user: {position: word}.
        :param max_N: Maximum steps to look forward (Eq. 12 from the "Interactive neural machine translation" paper).
        :param isles: Isles fixed by the user. List of (isle_index, [words]) (Although isle_index is never used).
        :param valid_next_words: List of candidate words to be the next one to generate (after generating fixed_words).
        :param idx2word: Mapping between indices and words.
        :return:
        """
        # Check input parameters and recover default values if needed
        if fixed_words is None:
            fixed_words = dict()
        if isles is None:
            isles = list()
        if idx2word is None:
            idx2word = dict()

        default_params = {'max_batch_size': 50,
                          'n_parallel_loaders': 8,
                          'beam_size': 5,
                          'normalize': False,
                          'mean_substraction': True,
                          'predict_on_sets': ['val'],
                          'maxlen': 20,
                          'n_samples': -1,
                          'model_inputs': ['source_text', 'state_below'],
                          'model_outputs': ['description'],
                          'dataset_inputs': ['source_text', 'state_below'],
                          'dataset_outputs': ['description'],
                          'sampling_type': 'max_likelihood',
                          'words_so_far': False,
                          'optimized_search': False,
                          'state_below_index': -1,
                          'output_text_index': 0,
                          'search_pruning': False,
                          'pos_unk': False,
                          'normalize_probs': False,
                          'alpha_factor': 0.0,
                          'coverage_penalty': False,
                          'length_penalty': False,
                          'length_norm_factor': 0.0,
                          'coverage_norm_factor': 0.0,
                          'output_max_length_depending_on_x': False,
                          'output_max_length_depending_on_x_factor': 3,
                          'output_min_length_depending_on_x': False,
                          'output_min_length_depending_on_x_factor': 2
                          }
        params = self.checkParameters(self.params, default_params)
        params['pad_on_batch'] = self.dataset.pad_on_batch[params['dataset_inputs'][-1]]

        # Prepare data generator: We won't use an Homogeneous_Data_Batch_Generator here
        X = dict()
        for input_id in params['model_inputs']:
            X[input_id] = src_sentence
        x = dict()
        for input_id in params['model_inputs']:
            x[input_id] = np.asarray([X[input_id]])
        samples, scores, alphas = self.interactive_beam_search(x,
                                                               params,
                                                               fixed_words=fixed_words,
                                                               max_N=max_N,
                                                               isles=isles,
                                                               valid_next_words=valid_next_words,
                                                               null_sym=self.dataset.extra_words['<null>'],
                                                               idx2word=idx2word)
        if params['length_penalty'] or params['coverage_penalty']:
            if params['length_penalty']:
                length_penalties = [((5 + len(sample)) ** params['length_norm_factor']
                                     / (5 + 1) ** params['length_norm_factor'])  # this 5 is a magic number by Google...
                                    for sample in samples]
            else:
                length_penalties = [1.0 for _ in len(samples)]

            if params['coverage_penalty']:
                coverage_penalties = []
                for k, sample in enumerate(samples):
                    # We assume that source sentences are at the first position of x
                    x_sentence = x[params['model_inputs'][0]][0]
                    alpha = np.asarray(alphas[k])
                    cp_penalty = 0.0
                    for cp_i in range(len(x_sentence)):
                        att_weight = 0.0
                        for cp_j in range(len(sample)):
                            att_weight += alpha[cp_j, cp_i]
                        cp_penalty += np.log(min(att_weight, 1.0))
                    coverage_penalties.append(params['coverage_norm_factor'] * cp_penalty)
            else:
                coverage_penalties = [0.0 for _ in len(samples)]
            scores = [co / lp + cp for co, lp, cp in zip(scores, length_penalties, coverage_penalties)]

        elif params['normalize_probs']:
            counts = [len(sample) ** params['alpha_factor'] for sample in samples]
            scores = [co / cn for co, cn in zip(scores, counts)]

        if self.n_best:
            n_best_indices = np.argsort(scores)
            n_best_scores = np.asarray(scores)[n_best_indices]
            n_best_samples = np.asarray(samples)[n_best_indices]
            if alphas is not None:
                n_best_alphas = [np.stack(alphas[i]) for i in n_best_indices]
            else:
                n_best_alphas = [None] * len(n_best_indices)
            n_best_list.append([n_best_samples, n_best_scores, n_best_alphas])

        best_score_idx = np.argmin(scores)
        best_sample = samples[best_score_idx]
        if self.return_alphas:
            best_alphas = np.asarray(alphas[best_score_idx])
        else:
            best_alphas = None
        if self.n_best:
            return (np.asarray(best_sample), scores[best_score_idx], np.asarray(best_alphas)), n_best_list
        else:
            return np.asarray(best_sample), scores[best_score_idx], np.asarray(best_alphas)

    @staticmethod
    def checkParameters(input_params, default_params):
        """
            Validates a set of input parameters and uses the default ones if not specified.
        """
        valid_params = [key for key in default_params]
        params = dict()

        # Check input parameters' validity
        for key, val in input_params.iteritems():
            if key in valid_params:
                params[key] = val

        # Use default parameters if not provided
        for key, default_val in default_params.iteritems():
            if key not in params:
                params[key] = default_val

        return params
