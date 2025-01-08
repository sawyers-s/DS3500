"""
assignta.py: Assign TAs to specific recitation sections using evolutionary computing
"""

from evo import Evo
import pandas as pd
import numpy as np
import random
from profiler import profile, Profiler
import csv


# define global variables

TA_DATA = pd.read_csv('tas.csv')
SECTION_DATA = pd.read_csv('sections.csv')


# define objective functions

@profile
def minimize_overallocation(solution):
    """ First objective function: Calculate overallocation penalty for each TA and sum overallocation
    penalties over all TAs. """
    # count labs assigned to each TA and calculate overallocation penalty (labs assigned - max_assigned) for each.
    # if penalty is negative, set equal to 0 (no penalty). sum penalties to get total penalty.
    return np.sum(np.maximum(solution.sum(axis = 1) - TA_DATA['max_assigned'].values, 0))

@profile
def minimize_conflicts(solution):
    """ Second objective function: Calculate the total number of TA time conflicts to minimize the number
    of TAs with one or more time conflicts. """
    # get assigned lab times for all TAs as well as each TA's id/row
    assigned_times = SECTION_DATA.loc[np.where(solution == 1)[1], 'daytime'].values
    ta_ids = np.where(solution == 1)[0]

    # use pandas to count conflicts by grouping by TA and checking for duplicate times (multiple conflicts count as 1
    # conflict --> use .any). sum TA conflicts to get total conflicts. (used ChatGPT for order of panda functions)
    return (pd.DataFrame({'TA_ID': ta_ids, 'Assigned_Time': assigned_times}).duplicated(['TA_ID', 'Assigned_Time'])
            .groupby(ta_ids).any().sum())

@profile
def minimize_undersupport(solution):
    """ Third objective function: Calculate the total number of undersupport penalty points to minimize
    the total penalty score across all sections. """
    # count number of TAs per section and calculate undersupport for each section (min_tas - assigned TAs). if penalty
    # is negative, set equal to 0 (no penalty). sum section penalties to get total penalty.
    return np.maximum(SECTION_DATA['min_ta'].values - solution.sum(axis = 0), 0).sum()

@profile
def minimize_unwilling(solution):
    """ Fourth objective function: Calculate the total number of times TAs are assigned to a section they are unwilling
    to support to minimize total unwilling instances across all sections. """
    # convert 'U' (unwilling) values to 1 and all other values to 0 (willing or preferred) for all TA rows. multiply
    # solution by unwilling values to compute number of unwilling assignments in result. sum for total instances.
    return np.sum(solution * (TA_DATA.iloc[:, 3:] == 'U').astype(int).values)

@profile
def minimize_unpreferred(solution):
    """ Fifth objective function: Calculate the total number of times TAs are assigned to a section they are willing
    (but not preferred) to support to minimize total unpreferred instances across all sections. """
    # extract willingness and preference from TA_DATA for all TA rows. sum total unpreferred assignments for each TA to
    # get total unpreferred assignments.
    return np.sum(solution * ((TA_DATA.iloc[:, 3:] == 'W') & (TA_DATA.iloc[:, 3:] != 'P')).values)


# define agents

@profile
def overallocation_minimizer(solutions):
    """ Agent to minimize total overallocation penalty. """
    # load max_assigned and TA preferences values
    max_support = TA_DATA['max_assigned'].values

    # choose a random solution from solutions
    solution = random.choice(solutions).copy()

    # calculate number of sections each TA is assigned to and number of TAs for each section
    sections_per_ta = np.sum(solution, axis = 1)
    tas_per_section = np.sum(solution, axis = 0)

    # identify overallocated TAs (assigned to more sections than their max)
    overallocated_tas = np.where(sections_per_ta > max_support)[0]

    # get preferences data
    unwilling_data = TA_DATA.iloc[:, 3:] == 'U'
    willing_data = TA_DATA.iloc[:, 3:] == 'W'

    for ta in overallocated_tas:
        # get sections TA is assigned to and remove them from a section if they are unwilling
        unwilling_sections = np.where(solution[ta, :] == 1)[0]
        for section in unwilling_sections:
            if unwilling_data.iloc[ta, section]:
                solution[ta, section] = 0
                sections_per_ta[ta] -= 1
                tas_per_section[section] -= 1

        # get sections TA is assigned to and remove them from a section if they are willing but not preferred
        willing_sections = np.where(solution[ta, :] == 1)[0]
        for section in willing_sections:
            if willing_data.iloc[ta, section] and not unwilling_data.iloc[ta, section]:
                solution[ta, section] = 0
                sections_per_ta[ta] -= 1
                tas_per_section[section] -= 1

        # keep removing TA from sections as long as they are overallocated
        while sections_per_ta[ta] > max_support[ta]:
            # find section with maximum TAs where this TA is assigned and remove TA
            assigned_sections = np.where(solution[ta, :] == 1)[0]
            section_to_remove = assigned_sections[np.argmax(tas_per_section[assigned_sections])]
            solution[ta, section_to_remove] = 0
            sections_per_ta[ta] -= 1
            tas_per_section[section_to_remove] -= 1

    return solution

@profile
def conflicts_minimizer(solutions):
    """ Agent to minimize time total TA time conflicts. """
    # choose a random solution from solutions
    solution = random.choice(solutions).copy()

    # get assigned labs and their TAs
    ta_indices, lab_indices = np.where(solution == 1)
    assigned_times = SECTION_DATA.loc[lab_indices, 'daytime'].values

    # create pandas dataframe to identify duplicates by TA and their assigned times (used ChatGPT for .duplicated syntax)
    is_conflicting = pd.Series(assigned_times).duplicated(keep = 'first').values

    # unassign TAs from extra (duplicate) lab times while keeping one at that time
    solution[ta_indices[is_conflicting], lab_indices[is_conflicting]] = 0

    return solution

@profile
def undersupport_minimizer(solutions):
    """ Agent to minimize total undersupport penalty. """
    # load min_ta and max_assigned values
    min_tas = SECTION_DATA['min_ta'].values
    max_support = TA_DATA['max_assigned'].values

    # choose a random solution from solutions
    solution = random.choice(solutions).copy()

    # calculate number of sections each TA is assigned to and number of TAs per section
    sections_per_ta = np.sum(solution, axis = 1)
    tas_per_section = np.sum(solution, axis = 0)

    # create list of overallocated sections (more than min_tas) and underallocated sections (fewer than min_tas)
    overallocated = np.where(tas_per_section > min_tas)[0]
    underallocated = np.where(tas_per_section < min_tas)[0]

    # identify overallocated TAs (assigned to more sections than their max)
    overallocated_tas = np.where(sections_per_ta > max_support)[0]

    # create list of all TAs that are available for movement (note: unassigned TAs, TAs in overallocated sections, or
    # TAs in unwilling sections are free to move)
    unwilling_data = TA_DATA.iloc[:, 3:] == 'U'
    available_tas = np.unique(np.concatenate([np.where(sections_per_ta == 0)[0],
                                              np.where((solution == 1) & unwilling_data.values)[0],
                                              np.where(np.sum(solution[:, overallocated], axis = 1) > 0)[0]]))

    # get preferred sections
    preferred_data = TA_DATA.iloc[:, 3:] == 'P'

    # move TAs from underallocated sections as needed
    for ta in available_tas:
        if ta in overallocated_tas or len(underallocated) == 0:
             # skip TAs that are overallocated or if there are no underallocated times
            continue

        # if preferred underallocated sections exist, choose one. otherwise use any underallocated section.
        preferred_underallocated_sections = underallocated[np.isin(underallocated, np.where(preferred_data.iloc[ta])[0])]
        target_section = (preferred_underallocated_sections[0] if len(preferred_underallocated_sections) > 0
                          else underallocated[0])

        # check if TA is currently assigned to any section
        assigned_sections = np.where(solution[ta] == 1)[0]
        if assigned_sections.size > 0:
            # reassign and update counts
            assigned_section = assigned_sections[0]
            solution[ta, assigned_section] = 0
            tas_per_section[assigned_section] -= 1

            # if overallocated section is now balanced, remove it from overallocated list
            if tas_per_section[assigned_section] == min_tas[assigned_section]:
                overallocated = overallocated[overallocated != assigned_section]

            # update target underallocated section counts
            solution[ta, target_section] = 1
            tas_per_section[target_section] += 1

            # if underallocated section is now balanced, remove it from underallocated list
            if tas_per_section[target_section] == min_tas[target_section]:
                underallocated = underallocated[underallocated != target_section]

    return solution

@profile
def unwilling_minimizer(solutions):
    """ Agent to minimize total unwilling instances across all sections. """
    # load TA preferences
    ta_preferences = TA_DATA.iloc[:, 3:].to_numpy()
    willing_sections = (ta_preferences == 'W')
    preferred_sections = (ta_preferences == 'P')

    # choose a random solution from solutions
    solution = random.choice(solutions).copy()

    # identify TAs assigned to unwilling sections
    ta_indices, section_indices = np.where(solution == 1)
    unwilling_assignments = (ta_preferences[ta_indices, section_indices] == 'U')

    # find indices of unwilling assignments and reassign TAs
    for idx in np.where(unwilling_assignments)[0]:
        ta = ta_indices[idx]
        section = section_indices[idx]

        # try to find a preferred section first (P) otherwise use willing section (W). use first P/W section.
        preferred_section_indices = np.where(preferred_sections[ta])[0]
        willing_section_indices = np.where(willing_sections[ta])[0]

        target_section = (preferred_section_indices[0] if preferred_section_indices.size > 0
                          else willing_section_indices[0] if willing_section_indices.size > 0 else None)

        if target_section is not None:
            # update solution (unassign from unwilling section, reassign to new section)
            solution[ta, section] = 0
            solution[ta, target_section] = 1

    return solution

@profile
def unpreferred_minimizer(solutions):
    """ Agent to minimize total unpreferred instances across all sections. """
    # load min_ta and TA preferences data
    min_tas = SECTION_DATA['min_ta'].values
    ta_preferences = TA_DATA.iloc[:, 3:].to_numpy()
    willing_sections = (ta_preferences == 'W')
    preferred_sections = (ta_preferences == 'P')

    # choose a random solution from solutions
    solution = random.choice(solutions).copy()

    # calculate number of TAs assigned to each section and identify unpreferred assignments
    tas_per_section = np.sum(solution, axis = 0)
    ta_indices, section_indices = np.where(solution == 1)
    unpreferred_indices = np.where(willing_sections[ta_indices, section_indices] &
                                ~preferred_sections[ta_indices, section_indices])[0]

    # identify undersupported preferred sections
    undersupported_preferred_sections = np.where((tas_per_section < min_tas) & preferred_sections.any(axis = 0))[0]

    # reallocate unpreferred assignments
    for idx in unpreferred_indices:
        ta, section = ta_indices[idx], section_indices[idx]

        # move TA to first undersupported preferred section. if none, move to any preferred section.
        preferred_ta_sections = np.where(preferred_sections[ta])[0]
        target_section = (undersupported_preferred_sections[0] if undersupported_preferred_sections.size > 0
                          else preferred_ta_sections[0] if preferred_ta_sections.size > 0 else None)

        if target_section is not None:
            solution[ta, section] = 0
            solution[ta, target_section] = 1
            tas_per_section[section] -= 1
            tas_per_section[target_section] += 1

    return solution

@profile
def shuffle_solutions(solutions):
    """ Agent to randomly shuffle TA assignments. """
    # choose a random solution from solutions and get shape
    solution = random.choice(solutions).copy()
    num_tas, num_sections = solution.shape

    # randomly select a shuffle ratio (e.g., between 0.1 and 0.3) to determine percentage of assignments to shuffle
    shuffle_ratio = random.uniform(0.1, 0.3)

    # calculate number of assignments to shuffle
    num_to_shuffle = int(num_tas * num_sections * shuffle_ratio)

    # randomly select indices to shuffle
    shuffle_indices = np.random.choice(num_tas * num_sections, num_to_shuffle, replace = False)

    # shuffle values (0 -> 1, 1 -> 0) for selected indices (used ChatGPT to help speed up operation using .flatten)
    solution_flat = solution.flatten()
    solution_flat[shuffle_indices] = 1 - solution_flat[shuffle_indices]

    # reshape back to original 2D form (used ChatGPT for .reshape as part of .flatten)
    return solution_flat.reshape(num_tas, num_sections)

@profile
def mutate_solutions(solutions):
    """ Agent to mutate TA assignments. """
    # choose a random solution from solutions
    solution = random.choice(solutions).copy()

    # set mutation rate and create mutation mask (True for mutation, False for no mutation) (used ChatGPT for syntax)
    mutation_rate = random.uniform(0.1, 0.3)
    mutation_mask = np.random.rand(*solution.shape) < mutation_rate

    # flip values in solution where mutation_mask is True (values to be mutated)
    solution[mutation_mask] = 1 - solution[mutation_mask]

    return solution


# define function to write csv file

def write_scores_csv(final_population, groupname, filename='nondominated_solution_scores.csv'):
    """ Write csv file containing summary table of objective scores for all non-dominated Pareto-optimal solutions.
     (used ChatGPT for csv-writing procedure) """
    # define list of expected objectives for csv header
    expected_objectives = ['overallocation', 'conflicts', 'undersupport', 'unwilling', 'unpreferred']

    with open(filename, mode = 'w', newline = '') as file:
        writer = csv.writer(file)
        # write header
        writer.writerow(['groupname'] + expected_objectives)

        # iterate over each non-dominated solution and write row with its objective scores
        for eval, sol in final_population.pop.items():
            # convert eval to dictionary and extract scores in the expected order
            eval_dict = dict(eval)
            writer.writerow([groupname] + [eval_dict.get(obj, None) for obj in expected_objectives])


# find non-dominated solutions using optimizer

def main():

    # create environment
    E = Evo()

    # add objective functions
    E.add_fitness_criteria('overallocation', minimize_overallocation)
    E.add_fitness_criteria('conflicts', minimize_conflicts)
    E.add_fitness_criteria('undersupport', minimize_undersupport)
    E.add_fitness_criteria('unwilling', minimize_unwilling)
    E.add_fitness_criteria('unpreferred', minimize_unpreferred)

    # register agents with Evo
    E.add_agent('overallocation_minimizer', overallocation_minimizer, k=1)
    E.add_agent('conflicts_minimizer', conflicts_minimizer, k=1)
    E.add_agent('undersupport_minimizer', undersupport_minimizer, k=1)
    E.add_agent('unwilling_minimizer', unwilling_minimizer, k=1)
    E.add_agent('unpreferred_minimizer', unpreferred_minimizer, k=1)
    E.add_agent('shuffle_solutions', shuffle_solutions, k=1)
    E.add_agent('mutate_solutions', mutate_solutions, k=1)

    # create an initial solution, S
    S = np.random.randint(2, size = (43, 17))
    E.add_solution(S)

    # run optimizer for five minutes (300 seconds) and print profiling report. also print initial and final populations.
    print('Initial population:\n', E)
    E.evolve(dom=100, status=10000, time_limit=300)
    print('Final population:\n', E)
    Profiler.report()

    # write objective scores of final population to csv file
    write_scores_csv(E, 'ssawyers')


if __name__ == '__main__':
    main()
