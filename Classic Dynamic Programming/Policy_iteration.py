#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created on 2020-05-24 11:28
@File:Policy_iteration.py
@Author:Zhuoli Yin
@Contact: yin195@purdue.edu
'''
import numpy as np
import gym

"""
Run agent in a deterministic environment, which could provide:
    (1) number of state
    (2) number of action
    (3) probability of transition given state and corresponding action, 
        and return tuple (probability, s', reward and done)
"""


class Agent:
    def __init__(self, env, discount_factor):
        self.env = env
        self.gamma = discount_factor

    def policyEvaluation(self, policy):
        """
        Do iterative policy evaluation for estimating V approx= v_pi
        :param policy:
        :return: array of value function
        """
        v = np.zeros(self.env.n_state)  # define an array to record value function of each state
        THETA = 0.001  # threshold to terminate, determining the accuracy of estimation
        DELTA = float('inf')

        while DELTA > THETA:  # policy evaluation terminates when delta < theta
            DELTA = 0

            for s in range(self.env.n_state):  # loop for each s in S, n_state is not currently defined in the env
                expected_value = 0  # V(s)

                for action, action_prob in enumerate(policy[s]):  # summation under all pi(s)
                    for prob, next_state, reward, done in self.env.P[s][action]:  # transition probs, P[s][a] == [(prob, s', r, done), ...]
                        expected_value += action_prob * prob * (reward + self.gamma * v[next_state])

                    DELTA = max(DELTA, np.abs(v[s] - expected_value))
                    v[s] = expected_value  # update value of each state

        return v

    def policyImprovement(self, s, v):
        """
        single policy improvement: update the best policy of a state for one iteration
        :param s: one specific state
        :param v: value function array
        :return: the "greedy" optimal action for the state
        """
        action_value = np.zeros(self.env.n_action)

        for a in range(self.env.n_action):
            for prob, next_state, reward, done in self.P[s][a]:
                action_value[a] += prob * (reward + self.gamma * v[next_state])

        return np.argmax(action_value), np.max(action_value)

    def optimize(self):
        """
        Policy iteration: run policy
        :return: optimal policy
        """
        policy = np.tile(np.eye(self.env.n_action)[1], (self.env.n_state, 1))  # build a table pi(a|s) and initialize

        policy_stable = False

        round_num = 0

        while not policy_stable:

            # policy stable = True

            print('\n Round Number:' + str(round_num))
            round_num += 1

            print('Current Policy')  # print the current optimal policy
            print(np.reshape([self.env.get_action_name[entry] for entry in [np.argmax(policy[s]) for s in range(self.env.n_state)]], self.env.shape))

            v = self.policyEvaluation(policy)
            print('Expected value according to Policy Evaluation')
            print(np.reshape(v, self.env.shape))  # print value of state in the form of environment shape

            for s in range(self.env.n_state):
                action_by_policy = np.argmax(policy[s])  # record old pi(s)
                best_action, best_action_value = self.policyImprovement(s, v)  # do one iteration of policy improvement
                policy[s] = np.eye(self.env.n_state)[best_action]

                if action_by_policy != best_action:  # judge if policy stable, else do policy evaluation
                    policy_stable = False

        policy = [np.argmax(policy[s]) for s in range(self.env.n_state)]  # return the optimal policy (which action to select) at each state

        return policy