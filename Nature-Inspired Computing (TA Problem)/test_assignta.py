"""
test_assignta.py: Unit test for each objective function to verify that objectives are working correctly
"""

import pytest
import pandas as pd
from assignta import (minimize_overallocation, minimize_conflicts, minimize_undersupport, minimize_unwilling,
                      minimize_unpreferred)


# define fixtures

@pytest.fixture
def solution_1():
    return pd.read_csv('test1.csv', header = None).to_numpy()

@pytest.fixture
def solution_2():
    return pd.read_csv('test2.csv', header = None).to_numpy()

@pytest.fixture
def solution_3():
    return pd.read_csv('test3.csv', header = None).to_numpy()


# write unit tests for each objective function

def test_minimize_overallocation(solution_1, solution_2, solution_3):
    # ensure all test solutions are loaded in correctly
    assert solution_1.shape == (43, 17), 'test1 not loaded in correctly'
    assert solution_2.shape == (43, 17), 'test2 not loaded in correctly'
    assert solution_3.shape == (43, 17), 'test3 not loaded in correctly'

    # ensure test solution overallocation penalty scores match expected scores on assignment pdf
    assert minimize_overallocation(solution_1) == 37, 'Actual overallocation did not match expected total for test1'
    assert minimize_overallocation(solution_2) == 41, 'Actual overallocation did not match expected total for test2'
    assert minimize_overallocation(solution_3) == 23, 'Actual overallocation did not match expected total for test3'

def test_minimize_conflicts(solution_1, solution_2, solution_3):
    # ensure all test solutions are loaded in correctly
    assert solution_1.shape == (43, 17), 'test1 not loaded in correctly'
    assert solution_2.shape == (43, 17), 'test2 not loaded in correctly'
    assert solution_3.shape == (43, 17), 'test3 not loaded in correctly'

    # ensure test solution time conflict penalty scores match expected scores on assignment pdf
    assert minimize_conflicts(solution_1) == 8, 'Actual time conflicts did not match expected total for test1'
    assert minimize_conflicts(solution_2) == 5, 'Actual time conflicts did not match expected total for test2'
    assert minimize_conflicts(solution_3) == 2, 'Actual time conflicts did not match expected total for test3'

def test_minimize_undersupport(solution_1, solution_2, solution_3):
    # ensure all test solutions are loaded in correctly
    assert solution_1.shape == (43, 17), 'test1 not loaded in correctly'
    assert solution_2.shape == (43, 17), 'test2 not loaded in correctly'
    assert solution_3.shape == (43, 17), 'test3 not loaded in correctly'

    # ensure test solution undersupport penalty scores match expected scores on assignment pdf
    assert minimize_undersupport(solution_1) == 1, 'Actual undersupport total did not match expected total for test1'
    assert minimize_undersupport(solution_2) == 0, 'Actual undersupport total did not match expected total for test2'
    assert minimize_undersupport(solution_3) == 7, 'Actual undersupport total did not match expected total for test3'

def test_minimize_unwilling(solution_1, solution_2, solution_3):
    # ensure all test solutions are loaded in correctly
    assert solution_1.shape == (43, 17), 'test1 not loaded in correctly'
    assert solution_2.shape == (43, 17), 'test2 not loaded in correctly'
    assert solution_3.shape == (43, 17), 'test3 not loaded in correctly'

    # ensure test solution unwilling penalty scores match expected scores on assignment pdf
    assert minimize_unwilling(solution_1) == 53, 'Actual unwilling total did not match expected total for test1'
    assert minimize_unwilling(solution_2) == 58, 'Actual unwilling total did not match expected total for test2'
    assert minimize_unwilling(solution_3) == 43, 'Actual unwilling total did not match expected total for test3'

def test_minimize_unpreferred(solution_1, solution_2, solution_3):
    # ensure all test solutions are loaded in correctly
    assert solution_1.shape == (43, 17), 'test1 not loaded in correctly'
    assert solution_2.shape == (43, 17), 'test2 not loaded in correctly'
    assert solution_3.shape == (43, 17), 'test3 not loaded in correctly'

    # ensure test solution unpreferred penalty scores match expected scores on assignment pdf
    assert minimize_unpreferred(solution_1) == 15, 'Actual unpreferred total did not match expected total for test1'
    assert minimize_unpreferred(solution_2) == 19, 'Actual unpreferred total did not match expected total for test2'
    assert minimize_unpreferred(solution_3) == 10, 'Actual unpreferred total did not match expected total for test3'
