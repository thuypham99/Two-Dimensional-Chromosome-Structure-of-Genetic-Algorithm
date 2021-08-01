# -*- coding: utf-8 -*-
"""
@author: Thuy Pham
"""

import math
from math import comb
from random import randrange
from time import perf_counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DAY_NO = 7
CROSS_PROB = 1.0
MUTATION_PROB = 0.33

seed = input(
    "Manually enter the random seed, leave this blank to randomly select the seed: "
)
if not seed:
    seed = randrange(1000000000)  # 1 billion posibilities
else:
    seed = int(seed)
print(f"Seed {seed} is used")
rng = np.random.default_rng(seed)


def inputInfomation():
    offWorkDaily = int(input("Enter the number of employee needed every working day: "))
    workingDayRequirement = int(input("Enter the number of days required: "))

    # probability of employee working on one day
    prob_offWorkDaily = workingDayRequirement / DAY_NO

    # maximum employees in rotation
    rawOffRotation = offWorkDaily / prob_offWorkDaily
    offRotation = math.ceil(offWorkDaily / prob_offWorkDaily)
    print(f"The offRotation is: {offRotation}")
    print("Please check the registration for day off in Excel")

    combination = comb(offRotation, offWorkDaily)
    print(f"Number of chromosome in population: {combination}")
    _ = input("Please wait... Press Enter to continue...")

    return (
        offWorkDaily,
        workingDayRequirement,
        rawOffRotation,
        offRotation,
        combination,
    )


def genPopulation(combination, offRotation, offWorkDaily):
    population = []
    # Create one chromosome
    initCondition = np.zeros((offRotation, 1))
    initCondition[:offWorkDaily] = 1
    for _ in range(combination):
        day = rng.permutation(initCondition)
        chromosome = day
        for _ in range(DAY_NO - 1):
            day = rng.permutation(initCondition)
            chromosome = np.concatenate([chromosome, day], axis=1)
        population.append(chromosome)
    population = np.array(population)
    return population


def getHardConstraintScore(population):
    # Contraint2 (check each row)
    sumRowChromosome = population.sum(axis=2)
    fitness2 = np.zeros_like(sumRowChromosome)
    criteria_2a = sumRowChromosome == workingDayRequirement
    fitness2[criteria_2a] = 20
    criteria_2b = sumRowChromosome == workingDayRequirement - 1
    criteria_2c = sumRowChromosome == workingDayRequirement + 1
    fitness2[criteria_2b | criteria_2c] = 8
    criteria_2d = sumRowChromosome == workingDayRequirement - 2
    criteria_2e = sumRowChromosome == workingDayRequirement + 2
    fitness2[criteria_2d | criteria_2e] = 4
    score2 = fitness2.sum(axis=1, keepdims=True)
    return score2


def getSoftConstraintScore(population):
    #find max consecutive day of each employee (each row represents one chromosome)
    groupConsecutive = np.zeros((population.shape[0],population.shape[1]))  
    for idx, groupMaxConsecutive in enumerate(population):
        # Append zeros columns at either sides of counts
        append = np.zeros((groupMaxConsecutive.shape[0],1),dtype=int)
        groupMaxConsecutive_ext = np.column_stack((append, groupMaxConsecutive, append))        
        # Get start and stop indices with 1s as triggers
        diffs = np.diff((groupMaxConsecutive_ext == 1).astype(int),axis=1)
        starts = np.argwhere(diffs == 1)
        stops = np.argwhere(diffs == -1)       
        # Get intervals using differences between start and stop indices
        distance = stops[:,1] - starts[:,1]       
        # Store intervals as a 2D array for further vectorized ops to make.
        #Count number of workingDay group occurence in each employee 
        count = np.bincount(starts[:,0])
        mask = np.arange(count.max()) < count[:,None]
        allConsecutive2D = mask.astype(float)
        allConsecutive2D[mask] = distance       
        # Get max along each row as final output
        result = allConsecutive2D.max(1)
        maxConsecutve = np.zeros(groupMaxConsecutive.shape[0])     
        #in case random full zero row of chromosome
        maxConsecutve[:result.shape[0]] = result
        groupConsecutive[idx] = maxConsecutve

    #Score
    fitness3 = np.zeros_like(groupConsecutive)
    criteria_3a = groupConsecutive == 2
    fitness3[criteria_3a] = 4
    score3 = fitness3.sum(axis=1, keepdims=True)
    return score3



def getTotalGoodness(population):
    score2 = getHardConstraintScore(population)
    score3 = getSoftConstraintScore(population)
    finalScore = score2 + score3
    return finalScore, score2, score3


def saveElitism(finalScore):
    bestFit = int(np.array(max(finalScore)))
    eliteIdx = np.where(finalScore == bestFit)[0][0]
    return eliteIdx, bestFit


# Termination condition for hard constraints
def hardRuleSatified(offRotation):
    bestHardRules = 20 * offRotation
    return bestHardRules


def nearHardRuleSatified(offRotation):
    nearBestHardRules = 20 * (offRotation - 1) + 4
    return nearBestHardRules


# Termination condition for soft constraints
def softRuleSatisfied(offRotation):
    bestSoftRules = 4 * offRotation
    return bestSoftRules


# Store the best fitness function
def totalRuleSatisfied(offRotation):
    bestTotalGoodness = 20 * offRotation + 4 * offRotation
    return bestTotalGoodness


def permutationGenerator(permutationIdx):
    N = len(permutationIdx)
    for x in range(N - 1):
        swapIdx = rng.integers(x + 1, N)
        permutationIdx[x], permutationIdx[swapIdx] = (
            permutationIdx[swapIdx],
            permutationIdx[x],
        )
    return permutationIdx


def unbiasedTournamentSelection(finalScore, chromosomeIdx):
    permuteInd1 = chromosomeIdx
    individual1 = finalScore[permuteInd1]

    permutationIdx = permutationGenerator(chromosomeIdx)
    permuteInd2 = permutationIdx
    individual2 = finalScore[permuteInd2]

    ind1Larger = individual1 > individual2
    ind2Larger = individual2 >= individual1

    ind1LargerIdx = permuteInd1[ind1Larger.flatten()]
    ind2LargerIdx = permuteInd2[ind2Larger.flatten()]
    indLargerIdx = np.concatenate([ind1LargerIdx, ind2LargerIdx])
    parentsIdx = indLargerIdx
    return parentsIdx


def crossover(parentsIdx, population):
    # using two-permute scheme when choosing crossover partner
    permutePar1 = parentsIdx
    parent1 = population[permutePar1]

    permutationIdx = permutationGenerator(parentsIdx)
    permutePar2 = permutationIdx
    parent2 = population[permutePar2]

    # select Offspring based on crossover rate
    # for each pair of parents
    crossoverOffspring = np.zeros_like(population)
    for idx, (offspring1, offspring2) in enumerate(zip(parent1, parent2)):
        # if random number < CROSS_PROB => swap each pair of parents
        if rng.random() < CROSS_PROB:
            crossoverLine = rng.choice(DAY_NO)  # random choose a crossover line
            # swap the starting point
            offspring1[:, crossoverLine:], offspring2[:, crossoverLine:] = (
                offspring2[:, crossoverLine:].copy(),
                offspring1[:, crossoverLine:].copy(),
            )
            # choose between Offspring 1 and Offspring 2 to become newOffspring
            chooseOffspring = rng.random() < 0.5
            crossoverOffspring[idx] = offspring1 if chooseOffspring else offspring2
        else:
            crossoverOffspring[idx] = offspring1
    return crossoverOffspring


def mutation(crossoverOffspring, mutatedColumn):
    mutationOffspring = np.zeros_like(crossoverOffspring)
    columnPool = crossoverOffspring.transpose(1, 0, 2).reshape(
        crossoverOffspring.shape[1], -1
    )
    for idx, newOffspring in enumerate(crossoverOffspring):
        random = rng.random()
        # if random numer < CROSS_PROB => mutate
        if random < MUTATION_PROB:
            overlayDayInd = rng.choice(
                newOffspring.shape[1], size=mutatedColumn, replace=False
            )
            replaceDayInd = rng.choice(
                columnPool.shape[1], size=mutatedColumn, replace=False
            )
            newOffspring[:, overlayDayInd] = columnPool[:, replaceDayInd]
            mutationOffspring[idx] = newOffspring
        else:
            mutationOffspring[idx] = newOffspring
    return mutationOffspring


def saveWorstSchedule(finalScore):
    worstFit = int(finalScore.min().item())
    worstScheduleIdx = np.where(finalScore == worstFit)[0][0]
    return worstScheduleIdx


def findNewGeneration(mutationOffspring, population, worstScheduleIdx, eliteIdx):
    newGeneration = mutationOffspring
    newGeneration[worstScheduleIdx] = population[eliteIdx]
    return newGeneration


def saveMaxTotalGoodnessOfGeneration(finalScore):
    maxGoodness = int(finalScore.max().item())
    maxTotalGoodnessIdx = np.where(finalScore == maxGoodness)[0][0]
    return maxTotalGoodnessIdx, maxGoodness


def saveMaxHardGeneration(score2):
    sameBestScore2Idx = np.where(score2 == score2.max())
    maxFinalSameBestScore2 = int(finalScore[sameBestScore2Idx].max().item())
    bestScore2Idx = np.where(finalScore == maxFinalSameBestScore2)[0][0]
    bestScore2 = int(score2[bestScore2Idx].item())
    return bestScore2, bestScore2Idx, maxFinalSameBestScore2


def saveMaxSoftGeneration(score3):
    bestScore3 = int(score3.max())
    # bestScore3Idx = np.where(score3 == bestScore3)[0][0]
    return bestScore3


def terminationCriteria(
    bestScore2,
    bestScore2Idx,
    bestHardRules,
    nearBestHardRules,
    iteration,
    maxGoodness,
    hardRulesNotYetSatisfied,
    done,
    stabilityCount,
):
    # bestScore2, bestScore2Idx, maxFinalSameBestScore2 = saveMaxHardGeneration(score2)
    # bestTotalGoodness = totalRuleSatisfied(offRotation)
    bestScore3 = saveMaxSoftGeneration(score3)
    # print(f"Best soft: {bestScore3}")

    if maxGoodness == bestTotalGoodness:
        print(f"All hard rules and soft rules satisfied at iteration: {iteration}")
        print(f"Max hard goodness: {bestScore2}")
        print(f"Max soft goodness: {bestScore3}")
        print(newGeneration[maxTotalGoodnessIdx])
        done = True
        # bestHardRules = hardRuleSatified(offRotation)
    if np.isclose(offRotation - rawOffRotation, 0):
        if hardRulesNotYetSatisfied and bestScore2 == bestHardRules:
            print(f"The first time all hard rules satisfied at iteration: {iteration}")
            print(f"Max hard goodness: {bestScore2}")
            print(newGeneration[bestScore2Idx])
            hardRulesNotYetSatisfied = False
    else:
        if hardRulesNotYetSatisfied and bestScore2 == nearBestHardRules:
            print(
                f"The first time all hard rules almost satisfied at iteration: {iteration}"
            )
            print(f"Max hard goodness: {bestScore2}")
            print(newGeneration[bestScore2Idx])
            hardRulesNotYetSatisfied = False

    mutatedColumn = 0
    if stabilityCount >= 70 and bestScore2 == bestHardRules:
        done = True
    if stabilityCount == 0:
        mutatedColumn += 1
    elif stabilityCount < 70:
        incrementCondition = 10 * math.ceil(stabilityCount / 10)
        mutatedColumn += math.ceil(stabilityCount / 10)
        if np.isclose(offRotation - rawOffRotation, 0):
            if (
                bestScore2 == bestHardRules
                and incrementCondition >= 10
                and incrementCondition % 10 == 0
            ):
                print(
                    f"All hard rules satisfied at iteration: {iteration}"
                    f" and stabilityCount: {incrementCondition}"
                )
        else:
            if (
                bestScore2 == nearBestHardRules
                and incrementCondition >= 10
                and incrementCondition % 10 == 0
            ):
                print(
                    f"All hard rules almost satisfied at iteration: {iteration}"
                    f" and stabilityCount: {incrementCondition}"
                )
            # print(newGeneration[bestScore2Idx])
    else:
        mutatedColumn = 7
    return mutatedColumn, hardRulesNotYetSatisfied, done


def displaySchedule(
    resultMaxGoodness,
    resultBestScore2,
    resultFinalScoreBestScore2,
):
    resultMaxGoodness = np.array(resultMaxGoodness)
    resultBestScore2 = np.array(resultBestScore2)
    resultFinalScoreBestScore2 = np.array(resultFinalScoreBestScore2)

    bestPossibleHardRule = int(resultBestScore2.max())
    if bestPossibleHardRule == bestHardRules:
        resultBestScore2Idx = np.where(resultBestScore2 == resultBestScore2.max())
        maxTotalGoodness = int(resultMaxGoodness[resultBestScore2Idx].max())
        print("Best schedule when all hard rules satisfied")
        print(f"Hard rule is {bestPossibleHardRule}")
        print(f"Final score is {maxTotalGoodness}")
        print(finalSchedule)
        print("================")
        excel_like_print(finalSchedule)
    else:
        resultBestScore2Idx = np.where(resultBestScore2 == resultBestScore2.max())
        maxFinalScoreBestScore2 = int(
            resultFinalScoreBestScore2[resultBestScore2Idx].max()
        )
        print("Best schedule when hard rules not yet satisfied")
        print(f"Hard Rule is {bestPossibleHardRule}")
        print(f"Final score is {maxFinalScoreBestScore2}")
        print(finalSchedule)
        print("================")
        excel_like_print(finalSchedule)
    return (
        bestPossibleHardRule,
    )


def excel_like_print(arr):
    print("\n".join(["\t".join(row.astype(int).astype(str)) for row in arr]))


def saveToExcel(sheets):
    with pd.ExcelWriter("GATwo.xlsx") as writer:
        for sheetName in sheets:
            sheets[sheetName].to_excel(writer, sheet_name=sheetName, index=False)


if __name__ == "__main__":
    # MAIN GA
    (
        offWorkDaily,
        workingDayRequirement,
        rawOffRotation,
        offRotation,
        combination,
    ) = inputInfomation()
    df = pd.read_excel("GATwo.xlsx", sheet_name=None)

    # Set initial population before loop to against effects
    population = genPopulation(combination, offRotation, offWorkDaily)
    chromosomeIdx = np.arange(population.shape[0])
    permutationIdx = chromosomeIdx.copy()

    # First fitness function
    finalScore, _, _ = getTotalGoodness(population)

    # set initial sabilityCount
    stabilityCount = 0
    mutatedColumn = 1

    # HardRuleSatisfied
    resultMaxGoodness = []

    # HardRuleNotSatisfied
    resultBestScore2 = []
    resultFinalScoreBestScore2 = []

    finalSchedule = None
    selectedScore2 = -1
    selectedMaxGoodness = -1
    selectedMaxFinalSameBestScore2 = -1
    previous = None
    hardRulesNotYetSatisfied = True
    done = False

    start = perf_counter()
    for iteration in range(200):  # (run 3 times => interation < 3)
        # xet them rieng score hard/rule cho terminaion
        eliteIdx, bestFit = saveElitism(finalScore)

        # Selection
        parentsIdx = unbiasedTournamentSelection(finalScore, chromosomeIdx)

        # Crossover
        crossoverOffspring = crossover(parentsIdx, population)

        # Mutation
        mutationOffspring = mutation(crossoverOffspring, mutatedColumn)

        # find worst schedule
        finalScore, _, _ = getTotalGoodness(mutationOffspring)
        worstScheduleIdx = saveWorstSchedule(finalScore)

        # NewGeneration
        newGeneration = findNewGeneration(
            mutationOffspring, population, worstScheduleIdx, eliteIdx
        )

        finalScore, score2, score3 = getTotalGoodness(newGeneration)

        # store result for displaying
        bestScore2, bestScore2Idx, maxFinalSameBestScore2 = saveMaxHardGeneration(
            score2
        )

        resultBestScore2.append(bestScore2)
        resultFinalScoreBestScore2.append(maxFinalSameBestScore2)

        maxTotalGoodnessIdx, maxGoodness = saveMaxTotalGoodnessOfGeneration(finalScore)
        bestTotalGoodness = totalRuleSatisfied(offRotation)
        resultMaxGoodness.append(maxGoodness)

        # StabilityCount
        if previous is not None and maxGoodness == previous:
            stabilityCount += 1
        else:
            stabilityCount = 0
        previous = maxGoodness

        bestHardRules = hardRuleSatified(offRotation)
        nearBestHardRules = nearHardRuleSatified(offRotation)
        
        # Check finalSchedule
        if bestScore2 > selectedScore2:
            selectedScore2 = bestScore2
            selectedMaxFinalSameBestScore2 = maxFinalSameBestScore2
            finalSchedule = newGeneration[bestScore2Idx]
        elif bestScore2 == selectedScore2:
            if maxFinalSameBestScore2 > selectedMaxFinalSameBestScore2:
                selectedMaxFinalSameBestScore2 = maxFinalSameBestScore2
                finalSchedule = newGeneration[bestScore2Idx]
                
        # termination here
        mutatedColumn, hardRulesNotYetSatisfied, done = terminationCriteria(
            bestScore2,
            bestScore2Idx,
            bestHardRules,
            nearBestHardRules,
            iteration,
            maxGoodness,
            hardRulesNotYetSatisfied,
            done,
            stabilityCount,
        )
        if done:
            break

        population = newGeneration

    print("==============")
    print("Number of generations =", iteration + 1)
    print("Running time (s) =",perf_counter() - start)
    (
        bestPossibleHardRule,
    ) = displaySchedule(
        resultMaxGoodness,
        resultBestScore2,
        resultFinalScoreBestScore2,
    )

    # Print plot
    lengthIteration = list(range(iteration + 1))

    # Save results to excel and text files
    results = pd.DataFrame(
        {
            "MaxGoodness": resultMaxGoodness,
            "BestScore2": resultBestScore2,
            "FinalScoreBestScore2": resultFinalScoreBestScore2,
            "iter": lengthIteration,
        },
    )
    df["results"] = results
    saveToExcel(df)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = "tab:blue"
    ax1.set_title("Best Hard Rule", fontsize=10)
    ax1.set_xlabel("Iteration", fontsize=10)
    ax1.set_ylabel("Best Hard Rule", fontsize=10)
    sns.lineplot(
        x=lengthIteration,
        y=resultBestScore2,
        sort=False,
        color=color,
        ax=ax1,
    )
    ax1.tick_params(axis="y")
    ax1.set_ylim(20, bestHardRules + 10)
    # plt.ylim(bottom=0)
    ax1.grid()

    if bestPossibleHardRule == bestHardRules:
        fig, ax2 = plt.subplots(figsize=(10, 6))
        color = "tab:red"
        ax2.set_title("Best Fitness Function", fontsize=10)
        ax2.set_xlabel("Iteration", fontsize=10)
        ax2.set_ylabel("Best Fitness Function", fontsize=10)
        sns.lineplot(
            x=lengthIteration,
            y=resultMaxGoodness,
            sort=False,
            color=color,
            ax=ax2,
        )
        ax2.tick_params(axis="y", color=color)
        ax2.set_ylim(20,bestTotalGoodness + 10)
        # plt.ylim(bottom=0)
        ax2.grid()
    else:
        fig, ax2 = plt.subplots(figsize=(10, 6))
        color = "tab:red"
        ax2.set_title("Final Score of Best Hard Rule", fontsize=10)
        ax2.set_xlabel("Iteration", fontsize=10)
        ax2.set_ylabel("Final Score of Best Hard Rule", fontsize=10)
        sns.lineplot(
            x=lengthIteration,
            y=resultFinalScoreBestScore2,
            sort=False,
            color=color,
            ax=ax2,
        )
        ax2.tick_params(axis="y", color=color)
        ax2.set_ylim(20,bestTotalGoodness + 10)
        # plt.ylim(bottom=0)
        ax2.grid()

        fig, ax3 = plt.subplots(figsize=(10, 6))
        color1 = "tab:green"
        color2 = "tab:purple"
        ax3.set_title(
            "Best Fitness Function and FinalScore of Best Hard Rule", fontsize=10
        )
        ax3.set_xlabel("Iteration", fontsize=10)
        ax3.set_ylabel("Best Fitness Function", fontsize=10)
        sns.lineplot(
            x=lengthIteration,
            y=resultMaxGoodness,
            sort=False,
            color=color1,
            ax=ax3,
        )
        sns.lineplot(
            x=lengthIteration,
            y=resultFinalScoreBestScore2,
            sort=False,
            color=color2,
            ax=ax3,
        )
        ax3.tick_params(axis="y", color=color)
        ax3.set_ylim(20,bestTotalGoodness + 10)
        # plt.ylim(bottom=0)
        ax3.grid()
        # fig, ax3 = plt.subplots(figsize=(50,20))
        # color ='tab:purple'
        # ax3.set_title('Best Fitness Function and FinalScore of Best Hard Rule', fontsize=16)
        # ax3.set_xlabel('Iteration', fontsize=16)
        # ax3.set_ylabel('Best Fitness Function', fontsize=16, color=color)
        # ax3 = sns.barplot(x = lengthIteration, y = resultMaxGoodness, palette='summer')
        # ax3 = sns.lineplot(x = lengthIteration, y = resultFinalScoreBestScore2, marker='o', sort=False, color=color)
        # ax3.tick_params(axis='y', color=color)
        # plt.ylim(top = bestTotalGoodness)
        # plt.ylim(bottom=0)
        # plt.grid()

    plt.show()

