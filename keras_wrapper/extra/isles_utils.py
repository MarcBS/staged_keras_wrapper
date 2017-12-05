#!/usr/bin/python
# encoding: utf-8
import numpy as np


def parse_input(line, dataset, idx2word):
    seqin = line.split()
    seqlen = len(seqin)
    seq = np.zeros(seqlen + 1, dtype='int64')
    seq_words = []
    for idx, sx in enumerate(seqin):
        if idx2word.get(sx) is not None:
            seq[idx] = idx2word.get(sx)
            seq_words.append(sx)
        else:
            seq[idx] = dataset.extra_words['<unk>']
            seq_words.append('<unk>')

    seq[-1] = dataset.extra_words['<pad>']
    seq_words.append('<pad>')
    return seq, seq_words


def longest_common_substring(s1, s2):
    """
    Gets the longest common substring between two strings
    :param s1:
    :param s2:
    :return:
    """
    m = [[0] * (1 + len(s2)) for _ in xrange(1 + len(s1))]
    longest, x_longest, y_longest = 0, 0, 0
    for x in xrange(1, 1 + len(s1)):
        for y in xrange(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
                    y_longest = y
            else:
                m[x][y] = 0
    return s1[x_longest - longest: x_longest], x_longest - longest, y_longest - longest


def common_prefix(s1, s2):
    """
    Gets the common prefix between two strings
    :param s1: String 1
    :param s2: String 2
    :return: Common prefix
    """
    for i in range(min(len(s1), len(s2))):
        if s1[i] != s2[i]:
            return i, s1[:i]
    return i + 1, s1[:i+1]


def longest_common_suffix(list1, list2):
    """
    Finds the longest common suffix in a list given another list.
    :param list1: List of suffix candidates
    :param list: List in which to search
    :return:
    """
    suffix_list = [list1[:i + 1] for i in range(len(list1))]
    suffix_list = sorted(suffix_list, key=len)
    suffix_list.reverse()
    # Find the longest common suffix
    for i, suffix in enumerate(suffix_list):
        if list2[-len(suffix):] == suffix:
            return suffix, -len(suffix)
    return [], 0

def find_isles(s1, s2, x=0, y=0):
    """
        Finds the set of isles of two lists of strings.
        :param s1: List of strings (words)
        :param s2: List of strings (words)
        :param x: Initial index of s1 (0 by default)
        :param y: Initial index of s2 (0 by default)
        :return: Duple of lists of duples: (list1, list2)
             list1 contains:
                [(i, [w1, ..., wn)], where:
                    i: Start index of the sequence [w1, ..., wn] in s1
             list2 is the same, but for s2
    """
    if s1 == [] or s2 == []:
        return [], []
    com, x_start, y_start = longest_common_substring(s1, s2)
    len_common = len(com)
    if len_common == 0:
        return [], []
    s1_bef = s1[:x_start]
    s1_aft = s1[x_start + len(com):]
    s2_bef = s2[:y_start]
    s2_aft = s2[y_start + len(com):]
    before = find_isles(s1_bef, s2_bef, x, y)
    after = find_isles(s1_aft, s2_aft, x + x_start + len_common, y + y_start + len_common)
    return (before[0] + [(x + x_start, com)] + after[0],
            before[1] + [(y + y_start, com)] + after[1])


def is_sublist(list1, list2):
    """
    Checks if list1 is included in list2
    :param list1:
    :param list2:
    :return:
    """
    return set(list2).issuperset(set(list1))


def subfinder(pattern, mylist):
    """
    Finds a pattern in a given list
    :param pattern:
    :param mylist:
    :return:
    """
    for start_pos in range(len(mylist)):
        if mylist[start_pos] == pattern[0] and mylist[start_pos:start_pos + len(pattern)] == pattern:
            return pattern, start_pos
    return [], -1


def compute_mouse_movements(isles, prev_isles, last_checked_index):
    """
    Computes the intersection between two sets of isles.
    The costs are:
        * 1 MA if we select an isle of size 1
        * 2 MA if we select an isle of size > 1
        * 0 MA if the isle was already selected
    (Note: Augmenting an isle already counts as MAs with the above criterion)

    :param isles: List of isles
    :param prev_isles: List of isles selected in the previous iteration
    :param last_checked_index: Don't compute already computed mouse actions
    :return: Number of mouse actions performed
    """
    mouse_actions_sentence = 0
    for index, words in isles:
        # For each isle, we check that it is not included in the previous isles
        if index > last_checked_index:  # We don't select validated prefixes
            if not any(map(lambda x: words[0] in x, prev_isles)):
                if len(words) > 1:
                    mouse_actions_sentence += 2  # Selection of new isle
                elif len(words) == 1:
                    mouse_actions_sentence += 1  # Isles of length 1 account for 1 mouse action
    return mouse_actions_sentence


def test_utils():

    s1 = 'Guia de Servicios de exploracion de red de CentreWare xi'
    s2 = 'Guia de instalaci\xc3\xb3n de Servicios de exploracion de red de CentreWare xi'

    print "Sentence 1:", s1
    print "Sentence 2:", s2
    print find_isles(s1.split(), s2.split())
